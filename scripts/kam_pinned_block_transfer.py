#!/usr/bin/env python3
"""Compare prefix protection and a matched positional control at the same block budget."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import load
from eval_kam_long_sparse import language_score, retrieval_score
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_backbone_block_attention import BLOCK, BUDGET, HEAD_DIM, QUERY_TILE, block_attention
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_query_window_flash import WindowAttention
from prepare_kam_attention_transfer import PROTOCOL, WIDTHS
from prepare_kam_evidence_fallback import locate
from prepare_kam_long_sparse import load_locked_tokenizer

PINS = {"prefix-2": (0, 1), "prefix-4": (0, 1, 2, 3), "late-4": (0, 8, 9, 10)}
RUNS = tuple((seed, arm) for seed in SEEDS for arm in PINS)


class PinAttention(WindowAttention):
    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        if self.mode not in PINS:
            return super().forward(module, q, k, v, mask, dropout, scaling, **kwargs)
        if mask is not None or dropout or not module.is_causal or q.size(-2) != k.size(-2):
            raise ValueError("Pinned blocks require strictly causal unpadded prefill")
        return block_attention(q, k, v, scaling, pinned=PINS[self.mode]), None


@torch.inference_mode()
def smoke():
    torch.manual_seed(827313)
    width = 1024
    def sample(h):
        return torch.randn(1, width, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    q, k, v = sample(8), sample(2), sample(2)
    positions = torch.arange(width, device="cuda")
    current = positions // BLOCK
    used = torch.arange(BUDGET, device="cuda") < current[:, None]+1
    check = {}
    for mode, pinned in PINS.items():
        out, routes = block_attention(q, k, v, pinned=pinned, record_routes=True)
        chosen = routes.index_select(-2, positions // QUERY_TILE).masked_fill(~used, -1)
        assert torch.equal(chosen[..., 0], current.expand_as(chosen[..., 0]))
        for index in pinned:
            assert ((chosen == index).any(dim=-1) | (current < index)).all()
        dup = (chosen[..., :, None] == chosen[..., None, :]).tril(-1) & used[..., :, None] & used[..., None, :]
        assert not dup.any()
        dense = ((chosen[..., None] == current).any(dim=-2)) & (positions <= positions[:, None])
        with sdpa_kernel(SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=dense,
                                                       enable_gqa=True).transpose(1, 2).bfloat16()
        torch.testing.assert_close(reference, out, atol=.012, rtol=.04)
        check[mode] = float((out.float()-reference.float()).abs().max())
    return check


def read(path):
    return json.loads(path.read_text())


def audit_positions(root, work):
    manifest_path = work / "transfer-manifest.json"
    manifest = read(manifest_path)
    prompts = []
    for width in WIDTHS:
        path = work / f"transfer-{width}.json"
        assert digest(path) == manifest["fixture"][str(width)]["sha256"]
        prompts.append(read(path)["retrieval"])
    tokenizer, tokenizer_path = load_locked_tokenizer(root)
    spans = locate(tokenizer, read(work / "retrieval.json"), prompts[0])
    counts = {mode: {"earlier_tokens": 0, "entire_earlier": 0, "complete_bridge": 0} for mode in PINS}
    for span, narrow, wide in zip(spans, *prompts, strict=True):
        assert (narrow["id"], narrow["expected"]) == (wide["id"], wide["expected"])
        stop = max(span["earlier"][1], (max(PINS["late-4"]) + 1) * BLOCK)
        assert narrow["ids"][:stop] == wide["ids"][:stop]
        earlier = range(*span["earlier"])
        bridge = range(*span["bridge"])
        for mode, pins in PINS.items():
            kept = sum(t // BLOCK in pins for t in earlier)
            counts[mode]["earlier_tokens"] += kept
            counts[mode]["entire_earlier"] += kept == len(earlier)
            counts[mode]["complete_bridge"] += all(t // BLOCK in pins for t in bridge)
    total = sum(end - start for start, end in (r["earlier"] for r in spans))
    write_json(work / "pinned-position-audit.json", {"manifest_sha256": digest(manifest_path),
        "tokenizer_sha256": digest(tokenizer_path), "presentations": len(spans),
        "earlier_tokens": total, "hard_pin_coverage": counts})


def run(root, source, work, mode, index):
    if mode == "audit":
        return audit_positions(root, work)
    if mode == "smoke":
        assert torch.cuda.get_device_name() == GPU
        print(smoke(), flush=True)
        return
    seed, variant = RUNS[index] if mode == "evaluate" else (SEEDS[0], "benchmark")
    model, stack, projection, _, checkpoint, attention = load(root, source, work, seed, PinAttention)
    manifest_path = work / "transfer-manifest.json"
    manifest = read(manifest_path)
    assert manifest["protocol"] == PROTOCOL
    outputs = {}
    with attention.installed():
        for width in WIDTHS:
            path = work / f"transfer-{width}.json"
            assert digest(path) == manifest["fixture"][str(width)]["sha256"]
            data = read(path)
            if mode == "evaluate":
                attention.mode = variant
                outputs[width] = {"pg19": language_score(model, stack, data, context=width),
                                  "retrieval": retrieval_score(model, stack, projection, data, context=width)}
            else:
                tokens = torch.tensor([data["pg19"][0]["windows"][0][:-1]], device="cuda")
                mask = torch.ones_like(tokens, dtype=torch.bool)
                def action(m):
                    attention.mode = m
                    return next_logits(model, stack, tokens, mask, projection=projection)[0]
                modes = ("flash", "protected", "tail-256", *PINS)
                outputs[width] = profile({m: lambda m=m: action(m) for m in modes}, repeats=15)
    path = work / (f"pinned-{seed}-{variant}.json" if mode == "evaluate" else "pinned-profile.json")
    write_json(path, {"seed": seed, "gpu": GPU, "checkpoint_sha256": checkpoint, "variant": variant,
                       "manifest_sha256": digest(manifest_path), "pins": PINS, "outcomes": outputs})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "work"):
        p.add_argument(name, type=Path)
    p.add_argument("mode", choices=("evaluate", "benchmark", "smoke", "audit"))
    p.add_argument("index", type=int, nargs="?", choices=range(len(RUNS)))
    a = p.parse_args()
    run(a.root, a.source, a.work, a.mode, a.index)
