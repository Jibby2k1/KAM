#!/usr/bin/env python3
"""Route annotated early evidence or equal-budget background through the same kernel."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import load
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_backbone_block_attention import BLOCK, BUDGET, HEAD_DIM, QUERY_TILE, block_attention
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_query_window_flash import WindowAttention
from prepare_kam_evidence_fallback import locate
from prepare_kam_long_sparse import load_locked_tokenizer

DOMAINS = ("musique", "hotpot")
WIDTHS = (4096, 8192)
VARIANTS = ("all-evidence", "all-background", "tail-evidence", "tail-background")
RUNS = tuple((d, s, v) for d in DOMAINS for s in SEEDS for v in VARIANTS)
PROTOCOL = "kam-evidence-pins-v1"


def read(path):
    return json.loads(path.read_text())


def inputs(work, domain, width):
    if domain == "musique":
        return read(work / f"transfer-{width}.json")["retrieval"]
    return read(work / f"hotpot-{width}.json")


def audit(root, work):
    tokenizer, tokenizer_path = load_locked_tokenizer(root)
    dataset = {}
    for domain, raw_name, manifest_name in (
            ("musique", "retrieval.json", "transfer-manifest.json"), ("hotpot", "native.json", "hotpot-manifest.json")):
        raw_path, manifest_path = work / raw_name, work / manifest_name
        spec = read(manifest_path)
        stem = "transfer" if domain == "musique" else "hotpot"
        for width in WIDTHS:
            assert digest(work / f"{stem}-{width}.json") == spec["fixture"][str(width)]["sha256"]
        narrow, wide = (inputs(work, domain, width) for width in WIDTHS)
        located = locate(tokenizer, read(raw_path), narrow)
        items = []
        for a, b, s in zip(narrow, wide, located, strict=True):
            assert (a["id"], a["family"], a["expected"]) == (b["id"], b["family"], b["expected"])
            begin, end = s["earlier"]
            # Block zero is already mandatory. Pin every other block overlapping the passage.
            lower, upper = max(1, begin // BLOCK), (end + BLOCK-1) // BLOCK
            count = upper - lower
            assert 0 <= count <= BUDGET - 3
            # Use the immediately following whole filler blocks and preserve the same slot count.
            control_end = upper + count
            assert control_end * BLOCK < min(s["gold"][0], s["foil"][0])
            assert a["ids"][:control_end * BLOCK] == b["ids"][:control_end * BLOCK]
            items.append({"id": a["id"], "family": a["family"], "expected": a["expected"], "earlier": [begin, end],
                          "bridge": s["bridge"], "evidence": [lower, upper], "background": [upper, control_end]})
        assert len(items) == 256
        for a, b in zip(items[::2], items[1::2], strict=True):
            assert a["id"] == b["id"] and [a["expected"], b["expected"]] == [0, 1]
            assert all(a[k] == b[k] for k in ("evidence", "background", "bridge", "earlier"))
        dataset[domain] = {"source_manifest_sha256": digest(manifest_path), "rows": items}
    write_json(work / "evidence-pin-plan.json", {"protocol": PROTOCOL, "tokenizer_sha256": digest(tokenizer_path),
                                                 "datasets": dataset})


class SpanAttention(WindowAttention):
    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        if self.mode not in VARIANTS:
            return super().forward(module, q, k, v, mask, dropout, scaling, **kwargs)
        if mask is not None or dropout or not module.is_causal or q.size(-2) != k.size(-2):
            raise ValueError("Evidence pinning requires strictly causal unpadded prefill")
        start = 0 if self.mode.startswith("all-") else q.size(-2) - 256
        return block_attention(q, k, v, scaling, keep_first=True, block_span=self.span, span_from=start), None


@torch.inference_mode()
def smoke():
    torch.manual_seed(273121)
    width, heads, kv = 1024, 8, 2

    def sample(h):
        return torch.randn(2, width, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)

    q, k, v = sample(heads), sample(kv), sample(kv)
    baseline = block_attention(q, k, v, keep_first=True)
    spans = torch.tensor([[1, 6], [4, 6]], device=q.device, dtype=torch.int32)
    zeros = torch.ones_like(spans)
    assert torch.equal(baseline, block_attention(q, k, v, keep_first=True, block_span=zeros))
    rows = torch.arange(width, device=q.device)
    current = rows // BLOCK
    used = torch.arange(BUDGET, device=q.device) < current[:, None] + 1
    results = {}
    for start in (0, 768):
        actual, route = block_attention(q, k, v, keep_first=True, block_span=spans, span_from=start, record_routes=True)
        assert torch.equal(baseline[:, :start], actual[:, :start])
        chosen = route.index_select(-2, rows // QUERY_TILE).masked_fill(~used, -1)
        assert not ((chosen[..., :, None] == chosen[..., None, :]).tril(-1) & used[..., :, None] & used[..., None, :]).any()
        assert ((chosen == 0).any(dim=-1)).all()
        for batch, (first, stop) in enumerate(spans.tolist()):
            for block in range(first, stop):
                assert ((chosen[batch] == block).any(dim=-1) | (current < block) | (rows < start)).all()
        visible = ((chosen[..., None] == current).any(dim=-2)) & (rows <= rows[:, None])
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=visible,
                                                      enable_gqa=True).transpose(1, 2).bfloat16()
        torch.testing.assert_close(actual, expected, atol=.012, rtol=.04)
        results[str(start)] = float((expected.float() - actual.float()).abs().max())
    return results


def configure(attention, case_rows):
    mode = attention.mode.split("-")[-1]
    attention.span = torch.tensor([row[mode] for row in case_rows], device="cuda", dtype=torch.int32)


@torch.inference_mode()
def run(root, source, work, mode, index):
    if mode == "audit":
        return audit(root, work)
    if mode == "smoke":
        assert torch.cuda.get_device_name() == GPU
        write_json(work / "evidence-pin-smoke.json", smoke())
        return
    domain, seed, variant = RUNS[index] if mode == "evaluate" else ("musique", SEEDS[0], "profile")
    plan_path = work / "evidence-pin-plan.json"
    plan = read(plan_path)
    assert plan["protocol"] == PROTOCOL
    model, stack, projection, _, checkpoint, attention = load(root, source, work, seed, SpanAttention)
    cases = plan["datasets"][domain]["rows"]
    fixture_name = "transfer" if domain == "musique" else "hotpot"
    fixture_path = work / f"{fixture_name}-manifest.json"
    fixture = read(fixture_path)
    assert plan["datasets"][domain]["source_manifest_sha256"] == digest(fixture_path)
    outcomes = {}
    with attention.installed():
        for width in WIDTHS:
            assert digest(work / f"{fixture_name}-{width}.json") == fixture["fixture"][str(width)]["sha256"]
            source_rows = inputs(work, domain, width)
            if mode == "evaluate":
                attention.mode = variant
                results = []
                for i in range(0, len(cases), 2):
                    pair, pin_rows = source_rows[i:i+2], cases[i:i+2]
                    labels = ("id", "expected", "family")
                    assert all(tuple(a[k] for k in labels) == tuple(b[k] for k in labels) for a,b in zip(pair, pin_rows, strict=True))
                    configure(attention, pin_rows)
                    ids = torch.tensor([r["ids"] for r in pair], device="cuda")
                    logits, _ = next_logits(model, stack, ids, torch.ones_like(ids, dtype=torch.bool), projection=projection)
                    for r, (a, b) in zip(pair, logits.tolist(), strict=True):
                        d = b-a
                        results.append({k: r[k] for k in labels} | {"margin": d,
                                        "correct": d != 0 and int(d > 0) == r["expected"]})
                outcomes[width] = results
            else:
                ids = torch.tensor([source_rows[0]["ids"]], device="cuda")
                mask = torch.ones_like(ids, dtype=torch.bool)
                spans = {kind: torch.tensor([cases[0][kind]], device=ids.device, dtype=torch.int32)
                         for kind in ("evidence", "background")}

                def action(arm):
                    attention.mode = arm
                    attention.span = spans.get(arm.split("-")[-1])
                    return next_logits(model, stack, ids, mask, projection=projection)[0]

                arms = ("flash", "protected", "tail-256", *VARIANTS)
                outcomes[width] = profile({arm: lambda arm=arm: action(arm) for arm in arms}, repeats=15)
    name = f"evidence-pins-{domain}-{seed}-{variant}.json" if mode == "evaluate" else "evidence-pin-profile.json"
    write_json(work / name, {"gpu": GPU, "domain": domain, "seed": seed, "variant": variant,
                           "plan_sha256": digest(plan_path), "checkpoint_sha256": checkpoint, "outcomes": outcomes})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "source", "work"):
        cli.add_argument(key, type=Path)
    cli.add_argument("mode", choices=("audit", "smoke", "benchmark", "evaluate"))
    cli.add_argument("index", nargs="?", type=int, choices=range(len(RUNS)))
    args = cli.parse_args()
    run(args.root, args.source, args.work, args.mode, args.index)
