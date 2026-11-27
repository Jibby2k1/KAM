#!/usr/bin/env python3
"""Score one registered preservation checkpoint on all five fixed test sets."""
import argparse
from pathlib import Path

import torch

from eval_kam_pretrained import EXPECTED, score_task
from kam_adapter_preservation import (
    FACTORIAL_RUNS, LOW_RATE_KL_RUNS, RUNS, digest, key, layers_for, load_registered_model, model_identity,
)
from kam_pretrained_adapters import AdapterStack


def load_checkpoint(root, seed, recipe, arm):
    model_id, revision, protocol = model_identity(root)
    ident = key(seed, recipe, arm)
    model, tokenizer = load_registered_model(root)
    stack = AdapterStack(model, arm, seed=seed, layers=layers_for(model))
    fingerprint = (root / "model-sha256.txt").read_text()
    identity = {}
    adapter_sha = None
    if arm != "pretrained":
        path = root / f"adapter-{ident}.pt"
        payload = torch.load(path, map_location="cuda", weights_only=True)
        identity = payload["identity"]
        assert (identity["seed"], identity["recipe"], identity["arm"]) == (seed, recipe, arm)
        assert identity["model_fingerprints"] == fingerprint and identity["revision"] == revision
        adapter_sha = digest(path)
        stack.modules.load_state_dict(payload["adapter"])
    stack.modules.eval()
    metadata = {
        "protocol": protocol, "arm": arm, "seed": seed, "recipe": recipe, "model": model_id,
        "revision": revision, "model_fingerprints": fingerprint, "adapter_sha256": adapter_sha,
        "new_trainable_parameters": sum(p.numel() for p in stack.modules.parameters()), "training": identity,
    }
    return model, tokenizer, stack, metadata


def evaluate(root, run):
    seed, recipe, arm = run
    model, tokenizer, stack, metadata = load_checkpoint(root, seed, recipe, arm)
    ident = key(seed, recipe, arm)
    for task in EXPECTED:
        result = root / f"{task}-{ident}.json"
        if not result.exists():
            score_task(root / "data", result, task, 64, model, tokenizer, stack, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--factorial", action="store_true")
    mode.add_argument("--low-rate-kl-pilot", action="store_true")
    a = parser.parse_args()
    if a.low_rate_kl_pilot:
        runs = LOW_RATE_KL_RUNS
    elif a.factorial:
        runs = FACTORIAL_RUNS
    else:
        runs = RUNS
    evaluate(a.root, runs[a.index])
