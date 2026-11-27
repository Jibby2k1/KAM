#!/usr/bin/env python3
"""Compare trained and shrunk residuals at a shared validation-only KL cap."""
import argparse
import json
from pathlib import Path

import torch

from eval_kam_adapter_preservation import load_checkpoint
from kam_adapter_preservation import FACTORIAL_TRAIN_RUNS, key
from probe_kam_adapter_strength import fixtures, predict
from train_kam_adapter_preservation import LENGTH, corpus, validation

KL_CAP = 0.01
BISECTION_STEPS = 11


def measure(root, index):
    seed, recipe, arm = FACTORIAL_TRAIN_RUNS[index]
    assert recipe in ("reference", "high_lr_kl")
    model, tokenizer, stack, metadata = load_checkpoint(root, seed, recipe, arm)
    checkpoint = key(seed, recipe, arm)
    up = [module.up.weight.detach().clone() for module in stack.modules]
    manifest = json.loads((root / "token-manifest.json").read_text())
    data = corpus(root, "validation", manifest)
    offsets = torch.arange(LENGTH + 1, device="cuda")
    grid = {}

    def sample(scale):
        with torch.no_grad():
            for module, weight in zip(stack.modules, up, strict=True):
                module.up.weight.copy_(weight * scale)
        grid[scale] = validation(model, stack, data, offsets)
        return grid[scale]["forward_kl"]

    lower, upper = 0.0, 1.0
    assert sample(lower) <= KL_CAP
    if sample(upper) <= KL_CAP:
        lower = upper
    else:
        for _ in range(BISECTION_STEPS):
            midpoint = (lower + upper) / 2
            if sample(midpoint) <= KL_CAP:
                lower = midpoint
            else:
                upper = midpoint
    assert lower == max(x for x, record in grid.items() if record["forward_kl"] <= KL_CAP)
    sample(lower)
    panels = fixtures(root, tokenizer)
    treated = {task: predict(model, stack, tokenizer, panel) for task, panel in panels.items()}
    stack.active = False
    reference = {task: predict(model, stack, tokenizer, panel) for task, panel in panels.items()}
    for panel in panels.values():
        del panel["encoded"]
    result = {
        **metadata, "gpu": torch.cuda.get_device_name(), "kl_cap": KL_CAP, "scale": lower,
        "sampled_validation": [{"scale": x, **record} for x, record in sorted(grid.items())],
        "panels": panels, "choices": treated, "untouched": reference,
    }
    path = root / f"matched-drift-{checkpoint}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, separators=(",", ":")))
    temp.replace(path)
    print(checkpoint, lower, grid[lower]["forward_kl"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    measure(args.root, args.index)
