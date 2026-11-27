#!/usr/bin/env python3
"""Score one registered preservation checkpoint on all five fixed test sets."""
import argparse
from pathlib import Path

import torch

from eval_kam_pretrained import EXPECTED, score_task
from kam_adapter_preservation import (
    MODEL_ID, PROTOCOL, REVISION, RUNS, digest, key, layers_for, load_registered_model,
)
from kam_pretrained_adapters import AdapterStack


def evaluate(root, index):
    seed, recipe, arm = RUNS[index]
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
        assert identity["model_fingerprints"] == fingerprint and identity["revision"] == REVISION
        adapter_sha = digest(path)
        stack.modules.load_state_dict(payload["adapter"])
    stack.modules.eval()
    metadata = {
        "protocol": PROTOCOL, "arm": arm, "seed": seed, "recipe": recipe, "model": MODEL_ID,
        "revision": REVISION, "model_fingerprints": fingerprint, "adapter_sha256": adapter_sha,
        "new_trainable_parameters": sum(p.numel() for p in stack.modules.parameters()), "training": identity,
    }
    for task in EXPECTED:
        result = root / f"{task}-{ident}.json"
        if not result.exists():
            score_task(root / "data", result, task, 64, model, tokenizer, stack, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int)
    a = parser.parse_args()
    evaluate(a.root, a.index)
