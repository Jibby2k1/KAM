#!/usr/bin/env python3
"""Audit dense attention mass and test a fixed-budget first-block intervention."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import BackboneAttention, WIDTHS, load, model_check
from eval_kam_long_sparse import language_score, retrieval_score
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_backbone_block_attention import BLOCK, BUDGET, HEAD_DIM, QUERY_TILE, block_attention, block_routes
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_pretrained_adapters import mixed_precision

DEPTHS = (8, 16, 32)
RUNS = tuple((seed, depth) for seed in SEEDS for depth in DEPTHS)
AUDIT_BOOKS = 8
AUDIT_QUESTIONS = 8
BINS = ("all", "pruned", "last_quarter", "final_query")
FIELDS = ("rows", "ordinary_mass_sum", "protected_mass_sum", "first_block_mass_sum", "omitted_rows",
          "omitted_first_mass_sum", "displaced_mass_sum", "ordinary_below_half", "protected_below_half")


def used_slots(rows):
    return torch.arange(BUDGET, device=rows.device) < (rows // BLOCK + 1)[:, None]


def check_routes(ordinary, protected, used):
    included = ((ordinary == 0) & used).any(dim=-1)
    assert ((protected == 0) & used).any(dim=-1).all()
    assert torch.equal(ordinary[..., :-1], protected[..., :-1])
    assert ((ordinary[..., -1] == protected[..., -1]) | ~used[..., -1] | ~included).all()
    assert ((protected[..., -1] == 0) | included).all()
    return included


def audit_mass(q, k, scaling):
    batch, heads, width, dim = q.shape
    kv = k.size(1)
    assert width == 4096 and scaling is not None
    rows = torch.cat((torch.arange(31, width, 64, device=q.device),
                      torch.tensor([width-1], device=q.device)))
    tile = rows // QUERY_TILE
    ordinary = block_routes(q, k).index_select(-2, tile)
    protected = block_routes(q, k, keep_first=True).index_select(-2, tile)
    valid = used_slots(rows)
    included = check_routes(ordinary, protected, valid)

    with torch.autocast("cuda", enabled=False):
        grouped = q.index_select(-2, rows).float().reshape(batch, kv, -1, dim)
        score = (grouped @ k.float().transpose(-1, -2)).reshape(batch, heads, len(rows), width) * scaling
        score = score.masked_fill(torch.arange(width, device=q.device) > rows[:, None], -torch.inf)
        dense_mass = score.softmax(dim=-1).unflatten(-1, (width // BLOCK, BLOCK)).sum(dim=-1)

    def kept(route):
        return (dense_mass.gather(-1, route.clamp_min(0).long()) * valid).sum(dim=-1)

    base_mass, protected_mass = kept(ordinary), kept(protected)
    first_mass = dense_mass[..., 0]
    omitted = ~included
    displaced = dense_mass.gather(-1, ordinary[..., -1:].clamp_min(0).long()).squeeze(-1)
    torch.testing.assert_close(protected_mass - base_mass, (first_mass-displaced) * omitted, rtol=0, atol=2e-6)
    regions = (rows >= 0, rows >= BLOCK*BUDGET, rows >= 3*width//4, rows == width-1)
    result = {}
    for name, region in zip(BINS, regions, strict=True):
        base, anchored, first = (item[..., region] for item in (base_mass, protected_mass, first_mass))
        missing = omitted[..., region]
        displaced_mass = displaced[..., region]
        result[name] = torch.stack((base.new_tensor(base.numel()), base.sum(), anchored.sum(), first.sum(),
            missing.sum(), (first*missing).sum(), (displaced_mass*missing).sum(),
            (base < .5).sum(), (anchored < .5).sum())).tolist()
    return result


class MassAttention(BackboneAttention):
    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        assert mask is None and dropout == 0.0 and module.is_causal and not self.selected
        assert module.layer_idx == len(self.records)
        self.records.append(audit_mass(q, k, scaling))
        return super().forward(module, q, k, v, mask, dropout=dropout, scaling=scaling, **kwargs)


@torch.inference_mode()
def audit(root, source, work, index):
    seed = SEEDS[index]
    model, stack, _, data, checkpoint, attention = load(root, source, work, seed, MassAttention)
    examples = ([("pg19", b["id"], None, b["windows"][0][:-1]) for b in data["pg19"][:AUDIT_BOOKS]]
                + [("musique", r["id"], r["expected"], r["ids"]) for r in data["retrieval"][:2*AUDIT_QUESTIONS]])
    observations = []
    with attention.installed(), mixed_precision():
        stack.context(None, False)
        for domain, ident, order, row in examples:
            attention.records = []
            model.model(input_ids=torch.tensor([row], device="cuda"), use_cache=False)
            assert len(attention.records) == 32
            observations.append({"domain": domain, "id": ident, "order": order, "mass": attention.records})
    write_json(work / f"mass-{seed}.json", {"seed": seed, "gpu": GPU, "checkpoint_sha256": checkpoint,
        "manifest_sha256": digest(work / "manifest.json"), "fields": FIELDS, "rows_per_prompt": 65,
        "observations": observations})


@torch.inference_mode()
def kernel_check():
    torch.manual_seed(190533)
    checks = {}
    for width in (512, 1024):
        def tensor(heads):
            return torch.randn(1, width, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        q, k, v = tensor(8), tensor(2), tensor(2)
        baseline, ordinary = block_attention(q, k, v, record_routes=True)
        output, protected = block_attention(q, k, v, keep_first=True, record_routes=True)
        rows = torch.arange(width, device=q.device)
        used = used_slots(rows)
        tile = rows // QUERY_TILE
        included = check_routes(ordinary.index_select(-2, tile), protected.index_select(-2, tile), used)
        if width == 512:
            assert bool(included.all()) and torch.equal(baseline, output)
        else:
            assert not bool(included.all())
        route = protected.index_select(-2, tile).masked_fill(~used, -1)
        mask = (route[..., None] == (rows // BLOCK)).any(dim=-2) & (rows <= rows[:, None])
        with sdpa_kernel(SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask,
                                                       enable_gqa=True).transpose(1, 2).bfloat16()
        torch.testing.assert_close(output, reference, rtol=.04, atol=.012)
        if width == 1024:
            suffix = 717
            for tensor_input in (q, k, v):
                tensor_input[..., suffix:, :] = 100
            after = block_attention(q, k, v, keep_first=True)
            assert torch.equal(output[:, :suffix], after[:, :suffix])
        checks[width] = {"intervened_head_rows": int((~included).sum()),
                         "max_abs_delta": float((output.float()-reference.float()).abs().max())}
    return checks


def benchmark(root, source, work):
    import triton

    seed = SEEDS[0]
    model, stack, proj, data, checkpoint, attention = load(root, source, work, seed)
    sample = [token for seq in data["pg19"][0]["windows"] for token in seq[:-1]]
    modes = {"flash": (0, False), **{f"{name}-{depth}": (depth, keep)
        for depth in DEPTHS for name, keep in (("ordinary", False), ("first", True))}}
    full = {}
    checks = kernel_check()
    with attention.installed():
        attention.keep_first = True
        integration = model_check(model, stack, proj, attention, data)
        for width in WIDTHS:
            tokens = torch.tensor([sample[:width]], device="cuda")
            mask = torch.ones_like(tokens, dtype=torch.bool)

            def action(depth, keep):
                attention.layers(depth)
                attention.keep_first = keep
                return next_logits(model, stack, tokens, mask, projection=proj)[0]

            for params in modes.values():
                assert torch.isfinite(action(*params)).all()
            full[width] = profile({name: lambda params=params: action(*params) for name, params in modes.items()},
                                  repeats=15)
    write_json(work / "first-profile.json", {"gpu": GPU, "torch": str(torch.__version__), "triton": triton.__version__,
        "manifest_sha256": digest(work / "manifest.json"), "seed": seed, "checkpoint_sha256": checkpoint,
        "checks": checks, "model_check": integration, "full_prefill": full})


def evaluate(root, source, work, index):
    seed, depth = RUNS[index]
    model, stack, proj, data, checkpoint, attention = load(root, source, work, seed)
    with attention.installed():
        attention.layers(depth)
        attention.keep_first = True
        payload = {"pg19": language_score(model, stack, data), "retrieval": retrieval_score(model, stack, proj, data)}
    write_json(work / f"first-{seed}-{depth}.json", {"seed": seed, "gpu": GPU, "depth": depth,
        "layers": sorted(attention.selected), "checkpoint_sha256": checkpoint,
        "manifest_sha256": digest(work / "manifest.json"), **payload})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "work"):
        p.add_argument(name, type=Path)
    p.add_argument("mode", choices=("smoke", "audit", "benchmark", "evaluate"))
    p.add_argument("index", type=int, nargs="?")
    a = p.parse_args()
    if a.mode == "smoke":
        assert torch.cuda.get_device_name() == GPU
        print(kernel_check(), flush=True)
        model, stack, proj, data, _, attention = load(a.root, a.source, a.work, SEEDS[0])
        with attention.installed():
            attention.keep_first = True
            print(model_check(model, stack, proj, attention, data), flush=True)
    elif a.mode == "audit":
        audit(a.root, a.source, a.work, a.index)
    elif a.mode == "benchmark":
        benchmark(a.root, a.source, a.work)
    else:
        evaluate(a.root, a.source, a.work, a.index)
