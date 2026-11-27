#!/usr/bin/env python3
"""Replay the five benchmark fixtures with and without exact TopK index reductions."""
import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from eval_kam_adapter_preservation import load_checkpoint
from eval_kam_pretrained import EXPECTED, score_task
from kam_adapter_preservation import key


def evaluate(root):
    seed, recipe, arm = 731221, "high_lr_kl", "kam"
    ident = key(seed, recipe, arm)
    model, tokenizer, stack, metadata = load_checkpoint(root, seed, recipe, arm)
    checks = {}
    with tempfile.TemporaryDirectory(prefix="route-benchmark-", dir=root) as temporary:
        for index, (task, n) in enumerate(EXPECTED.items()):
            official = json.loads((root / f"{task}-{ident}.json").read_text())
            modes = ("full_validation", "shape_validation")
            if index % 2:
                modes = modes[::-1]
            outputs, elapsed = {}, {}
            for mode in modes:
                for module in stack.modules:
                    module.router.validate_generated_indices = mode == "full_validation"
                path = Path(temporary) / f"{task}-{mode}.json"
                torch.cuda.synchronize()
                started = time.perf_counter()
                score_task(root / "data", path, task, 64, model, tokenizer, stack, metadata)
                torch.cuda.synchronize()
                elapsed[mode] = time.perf_counter() - started
                data = json.loads(path.read_text())
                for field in ("adapter_sha256", "model_fingerprints", "source_sha256", "context_tokens", "gpu"):
                    assert data["metadata"][field] == official["metadata"][field], (task, field)
                outputs[mode] = data["records"]
            verified, optimized = (outputs[name] for name in ("full_validation", "shape_validation"))
            assert len(verified) == len(optimized) == len(official["records"]) == n
            checks[task] = {"n": n, "order": modes, "seconds": elapsed,
                            "optimized_row_mismatches": sum(a != b for a, b in zip(verified, optimized, strict=True)),
                            "original_row_mismatches": sum(a != b for a, b in zip(verified, official["records"], strict=True))}
            print(metadata["model"], task, checks[task], flush=True)

    result = {"checkpoint": metadata, "gpu": torch.cuda.get_device_name(), "tasks": checks}
    output = root / "benchmark-route-validation.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    evaluate(args.root)
