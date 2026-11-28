#!/usr/bin/env python3
"""Test the fixed final-window policy on native HotpotQA at two lengths."""
import argparse
import json
from pathlib import Path

import torch

from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import load
from eval_kam_long_sparse import retrieval_score
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_query_window_flash import WindowAttention
from prepare_kam_long_hotpot_transfer import PROTOCOL, WIDTHS

ARMS = ("flash", "protected", "tail-256")
RUNS = tuple((s, m) for s in SEEDS for m in ARMS)


def read(path):
    return json.loads(path.read_text())


def evaluate(root, source, work, mode, index):
    seed, arm = RUNS[index] if mode == "evaluate" else (SEEDS[0], "profile")
    fixture_path = work / "hotpot-manifest.json"
    fixture = read(fixture_path)
    assert fixture["protocol"] == PROTOCOL and fixture["tokenizer_sha256"] == digest(root/"model"/"tokenizer.json")
    model, stack, projection, _, checkpoint, attention = load(root, source, work, seed, WindowAttention)
    baseline = {"seed": seed, "gpu": GPU, "arm": arm, "checkpoint_sha256": checkpoint,
                "hotpot_manifest_sha256": digest(fixture_path)}
    outcomes = {}
    with attention.installed():
        for width in WIDTHS:
            path = work / f"hotpot-{width}.json"
            assert digest(path) == fixture["fixture"][str(width)]["sha256"]
            examples = read(path)
            if mode == "evaluate":
                attention.mode = arm
                outcomes[width] = retrieval_score(model, stack, projection, {"retrieval": examples}, context=width)
            else:
                tokens = torch.tensor([examples[0]["ids"]], device="cuda")
                mask = torch.ones_like(tokens, dtype=torch.bool)

                def action(active):
                    attention.mode = active
                    return next_logits(model, stack, tokens, mask, projection=projection)[0]

                outcomes[width] = profile({name: lambda name=name: action(name) for name in ARMS}, repeats=15)
    name = f"hotpot-{seed}-{arm}.json" if mode == "evaluate" else "hotpot-profile.json"
    write_json(work / name, baseline | {"outcomes": outcomes})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "work"):
        p.add_argument(name, type=Path)
    p.add_argument("mode", choices=("evaluate", "benchmark"))
    p.add_argument("index", type=int, nargs="?", choices=range(len(RUNS)))
    a = p.parse_args()
    evaluate(a.root, a.source, a.work, a.mode, a.index)
