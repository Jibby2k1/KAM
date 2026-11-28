#!/usr/bin/env python3
"""Confirm the locked query-window arm on independent matched 4K and 8K panels."""
import argparse
import json
from pathlib import Path

import torch

from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import load
from eval_kam_long_sparse import language_score, retrieval_score
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_query_window_flash import WindowAttention
from prepare_kam_attention_transfer import PROTOCOL, WIDTHS

ARMS = ("flash", "protected", "selected")
RUNS = tuple((seed, mode) for seed in SEEDS for mode in ARMS)


def read(path):
    return json.loads(path.read_text())


def run(root, source, work, mode, index):
    plan_path = work / "routing-selection.json"
    plan = read(plan_path)
    selected = plan["transfer_arm"]
    assert selected in ("tail-32", "tail-256")
    fixture_path = work / "transfer-manifest.json"
    fixture = read(fixture_path)
    assert fixture["protocol"] == PROTOCOL
    assert fixture["tokenizer_sha256"] == digest(root / "model" / "tokenizer.json")
    seed, arm = RUNS[index] if mode == "evaluate" else (SEEDS[0], None)
    model, stack, projection, _, checkpoint, attention = load(root, source, work, seed, WindowAttention)
    common = {"seed": seed, "gpu": GPU, "checkpoint_sha256": checkpoint,
              "transfer_manifest_sha256": digest(fixture_path), "selection_sha256": digest(plan_path)}
    outcomes = {}
    with attention.installed():
        for width in WIDTHS:
            path = work / f"transfer-{width}.json"
            assert digest(path) == fixture["fixture"][str(width)]["sha256"]
            data = read(path)
            if mode == "evaluate":
                attention.mode = selected if arm == "selected" else arm
                outcomes[width] = {"pg19": language_score(model, stack, data, context=width),
                                   "retrieval": retrieval_score(model, stack, projection, data, context=width)}
            else:
                tokens = torch.tensor([data["pg19"][0]["windows"][0][:-1]], device="cuda")
                mask = torch.ones_like(tokens, dtype=torch.bool)

                def action(variant):
                    attention.mode = selected if variant == "selected" else variant
                    return next_logits(model, stack, tokens, mask, projection=projection)[0]

                outcomes[width] = profile({v: lambda v=v: action(v) for v in ARMS}, repeats=15)
    if mode == "evaluate":
        write_json(work / f"transfer-{seed}-{arm}.json", common | {"arm": arm, "selected": selected, "outcomes": outcomes})
    else:
        write_json(work / "transfer-profile.json", common | {"selected": selected, "full_prefill": outcomes})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "work"):
        p.add_argument(name, type=Path)
    p.add_argument("mode", choices=("evaluate", "benchmark"))
    p.add_argument("index", type=int, nargs="?", choices=range(len(RUNS)))
    a = p.parse_args()
    run(a.root, a.source, a.work, a.mode, a.index)
