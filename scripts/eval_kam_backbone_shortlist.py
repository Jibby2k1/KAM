#!/usr/bin/env python3
"""Replace Llama attention at fixed depths and replay the locked holdout."""
import argparse
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS

from benchmark_kam_sparse_live_read import profile
from eval_kam_long_sparse import language_score, prepare_run, retrieval_score
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_backbone_block_attention import BLOCK, BUDGET, GROUP, HEAD_DIM, QUERY_TILE, block_attention
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_pretrained_adapters import mixed_precision

DEPTHS = (0, 8, 16, 32)
RUNS = tuple((seed, depth) for seed in SEEDS for depth in DEPTHS)
WIDTHS = (512, 2048, 4096, 8192)
ARM = "top4-linear"


class BackboneAttention:
    def __init__(self, model):
        assert model.config.num_hidden_layers == 32 and model.config._attn_implementation == "sdpa"
        assert model.config.num_attention_heads == 32 and model.config.num_key_value_heads == 8
        self.original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.selected = frozenset()

    def layers(self, depth):
        self.selected = frozenset(range(32 // depth - 1, 32, 32 // depth)) if depth else frozenset()

    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        if module.layer_idx not in self.selected:
            return self.original(module, q, k, v, mask, dropout=dropout, scaling=scaling, **kwargs)
        if mask is not None or dropout != 0.0 or not module.is_causal or q.size(-2) != k.size(-2):
            raise ValueError("Shortlisted backbone attention requires unpadded, uncached causal prefill")
        return block_attention(q, k, v, scaling), None

    @contextmanager
    def installed(self):
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.forward
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                yield
        finally:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = self.original


@torch.inference_mode()
def kernel_checks():
    torch.manual_seed(731213)
    results = {}
    for batch, heads, kv, width in ((1, 32, 8, 103), (2, 8, 2, 512), (1, 8, 2, 777)):
        def tensor(n):
            return torch.randn(batch, width, n, HEAD_DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        q, k, v = tensor(heads), tensor(kv), tensor(kv)
        out, route = block_attention(q, k, v, record_routes=True)
        tile_starts = torch.arange(0, width, QUERY_TILE, device=q.device)
        pos = tile_starts // BLOCK
        assert torch.equal(route[..., 0], pos.expand_as(route[..., 0]))
        previous = torch.minimum(pos + 1, torch.full_like(pos, BUDGET))
        used = torch.arange(BUDGET, device=q.device) < previous[:, None]
        assert ((route >= 0) | ~used).all()
        assert ((route <= pos[:, None]) | ~used).all()
        valid_routes = route.masked_fill(~used, -1)
        same = valid_routes[..., :, None] == valid_routes[..., None, :]
        assert not (same.tril(-1) & used[..., :, None] & used[..., None, :]).any()
        if width <= BUDGET * BLOCK:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                reference = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            # Rebuild the sparse mask and softmax independently with Torch.
            rows = torch.arange(width, device=q.device)
            rows_route = valid_routes.index_select(-2, rows // QUERY_TILE)
            allowed = (rows_route[..., None] == (rows // BLOCK)).any(dim=-2) & (rows <= rows[:, None])
            with sdpa_kernel(SDPBackend.MATH):
                reference = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=allowed,
                                                           enable_gqa=True).bfloat16()
            suffix = 617
            for t in (q, k, v):
                t[..., suffix:, :] = 100
            changed, altered_routes = block_attention(q, k, v, record_routes=True)
            assert torch.equal(out[:, :suffix], changed[:, :suffix])
            assert torch.equal(route[..., :triton_tiles(suffix), :], altered_routes[..., :triton_tiles(suffix), :])
        expected = reference.transpose(1, 2)
        torch.testing.assert_close(out, expected, rtol=.04, atol=.012)
        results[str(width)] = {"max_abs_delta": float((out.float()-expected.float()).abs().max()),
                               "shape": list(out.shape), "support": "full" if width <= BUDGET * BLOCK else "shortlist"}
    return results


def triton_tiles(width):
    return (width + QUERY_TILE - 1) // QUERY_TILE


@torch.inference_mode()
def wide_kernel_check(work):
    """Verify the 8K dot-product tiling used after the shared-memory fix."""
    torch.manual_seed(731239)
    width, heads, kv = 8192, 32, 8

    def tensor(n):
        return torch.randn(1, width, n, HEAD_DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)

    q, k, v = tensor(heads), tensor(kv), tensor(kv)
    out, route = block_attention(q, k, v, record_routes=True)
    assert torch.isfinite(out).all()
    keys = k.repeat_interleave(heads // kv, dim=1)
    values = v.repeat_interleave(heads // kv, dim=1)
    means = keys.reshape(1, heads, width // GROUP, GROUP, HEAD_DIM).float().mean(dim=-2).bfloat16()
    block_ids = torch.arange(width // BLOCK, device=q.device)
    distances = {}
    for start in (4096, 6144, 8160):
        current = start // BLOCK
        proxies = q[:, :, start:start+1].float() @ means.float().transpose(-1, -2)
        coarse = proxies.reshape(1, heads, width // BLOCK, BLOCK // GROUP).amax(dim=-1)
        coarse = coarse.masked_fill(block_ids >= current - 1, -torch.inf)
        older = coarse.argsort(dim=-1, descending=True, stable=True)[..., :BUDGET-2]
        local = torch.tensor([current, current-1], device=q.device).expand(1, heads, 2)
        expected_route = torch.cat((local, older), dim=-1)
        assert torch.equal(route[:, :, start//QUERY_TILE], expected_route)
        key_ids = (expected_route[..., None] * BLOCK + torch.arange(BLOCK, device=q.device)).flatten(-2)
        gather = key_ids[..., None].expand(1, heads, BUDGET*BLOCK, HEAD_DIM)
        retained_k, retained_v = (t.gather(dim=2, index=gather).float() for t in (keys, values))
        select = torch.tensor([start, start+15, start+31], device=q.device)
        visible = key_ids[:, :, None, :] <= select[:, None]
        with sdpa_kernel(SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(q.index_select(2, select).float(), retained_k, retained_v,
                                                       attn_mask=visible).bfloat16()
        expected = reference.transpose(1, 2)
        actual = out.index_select(1, select)
        torch.testing.assert_close(actual, expected, rtol=.04, atol=.012)
        distances[start] = float((actual.float()-expected.float()).abs().max())
    write_json(work / "wide-check.json", {"gpu": torch.cuda.get_device_name(), "width": width,
        "heads": heads, "kv": kv, "route_checks": heads*len(distances), "max_abs_delta": distances,
        "kernel_sha256": digest(Path(__file__).with_name("kam_backbone_block_attention.py"))})


@torch.inference_mode()
def model_check(model, stack, proj, attention, data):
    tokens = torch.tensor([book["windows"][0][:513] for book in data["pg19"][:2]], device="cuda")
    prefix = tokens[:, :-1]
    mask = torch.ones_like(prefix, dtype=torch.bool)
    outputs = {}
    for depth in (0, 32):
        attention.layers(depth)
        stack.context(None, False)
        with mixed_precision():
            hidden = model.model(input_ids=prefix, use_cache=False).last_hidden_state
            pred = model.lm_head(hidden[:, -64:]).float()
        nll = F.cross_entropy(pred.transpose(1, 2), tokens[:, -64:], reduction="none").mean(dim=-1).tolist()
        logits, _ = next_logits(model, stack, prefix, mask, projection=proj)
        outputs[depth] = nll, logits
    change = [b-a for a, b in zip(outputs[0][0], outputs[32][0], strict=True)]
    assert max(map(abs, change)) < .015, change
    return {"all_layer_full_prefix_nll_change": change,
            "max_abs_label_logit_delta": float((outputs[0][1]-outputs[32][1]).abs().max())}


def load(root, source, work, seed):
    model, stack, proj, data, check = prepare_run(root, source, work, seed, ARM)
    stack.backend("sparse")
    return model, stack, proj, data, check, BackboneAttention(model)


def benchmark(root, source, work):
    import transformers
    import triton

    checks = kernel_checks()
    seed = SEEDS[0]
    model, stack, proj, data, checkpoint, attention = load(root, source, work, seed)
    sample = [token for seq in data["pg19"][0]["windows"] for token in seq[:-1]]
    full, layers = {}, {}
    with attention.installed():
        model_checks = model_check(model, stack, proj, attention, data)
        for width in WIDTHS:
            tokens = torch.tensor([sample[:width]], device="cuda")
            mask = torch.ones_like(tokens, dtype=torch.bool)

            def action(depth):
                attention.layers(depth)
                return next_logits(model, stack, tokens, mask, projection=proj)[0]

            full[str(width)] = profile({str(depth): lambda depth=depth: action(depth) for depth in DEPTHS}, repeats=15)
        for depth in DEPTHS:
            attention.layers(depth)
            layers[depth] = sorted(attention.selected)
    write_json(work / "backbone-profile.json", {"gpu": GPU, "torch": str(torch.__version__), "triton": triton.__version__,
               "transformers": transformers.__version__, "seed": seed, "checkpoint_sha256": checkpoint,
               "manifest_sha256": digest(work / "manifest.json"), "checks": checks, "model_check": model_checks,
               "depth_layers": layers, "full_prefill": full, "block": BLOCK, "group": GROUP,
               "query_tile": QUERY_TILE, "budget": BUDGET})


def evaluate(root, source, work, index):
    seed, depth = RUNS[index]
    model, stack, proj, data, checkpoint, attention = load(root, source, work, seed)
    with attention.installed():
        attention.layers(depth)
        payload = {"pg19": language_score(model, stack, data),
                   "retrieval": retrieval_score(model, stack, proj, data)}
    write_json(work / f"backbone-{seed}-{depth}.json", {"gpu": GPU, "seed": seed, "depth": depth,
               "layers": sorted(attention.selected), "manifest_sha256": digest(work / "manifest.json"),
               "checkpoint_sha256": checkpoint, **payload})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "work"):
        p.add_argument(name, type=Path)
    p.add_argument("mode", choices=("smoke", "wide-check", "benchmark", "evaluate"))
    p.add_argument("index", type=int, nargs="?", choices=range(len(RUNS)))
    args = p.parse_args()
    if args.mode == "smoke":
        assert torch.cuda.get_device_name() == GPU
        print(kernel_checks(), flush=True)
        model, stack, proj, data, _, attention = load(args.root, args.source, args.work, SEEDS[0])
        with attention.installed():
            print(model_check(model, stack, proj, attention, data), flush=True)
    elif args.mode == "wide-check":
        assert torch.cuda.get_device_name() == GPU
        wide_kernel_check(args.work)
    elif args.mode == "benchmark":
        benchmark(args.root, args.source, args.work)
    else:
        evaluate(args.root, args.source, args.work, args.index)
