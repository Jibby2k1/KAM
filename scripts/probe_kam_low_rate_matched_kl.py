#!/usr/bin/env python3
"""Measure GSM8K after shrinking low-rate dot to the paired KAM validation KL."""
import argparse
import json
from pathlib import Path

import torch

from eval_kam_adapter_preservation import load_checkpoint
from eval_kam_pretrained import score_task
from kam_adapter_preservation import SEEDS, key
from probe_kam_matched_drift import largest_scale_under_cap
from train_kam_adapter_preservation import LENGTH, corpus, validation


def measure(root, index, layer_halves):
    seed = SEEDS[index]
    recipe = "lower_lr"
    manifest = json.loads((root / "token-manifest.json").read_text())
    data = corpus(root, "validation", manifest)
    offsets = torch.arange(LENGTH + 1, device="cuda")
    model, tokenizer, kam, kam_metadata = load_checkpoint(root, seed, recipe, "kam")
    target = validation(model, kam, data, offsets)["forward_kl"]
    assert target > 0
    kam.close()
    del model, tokenizer, kam
    torch.cuda.empty_cache()

    model, tokenizer, dot, metadata = load_checkpoint(root, seed, recipe, "dot")
    original = [module.up.weight.detach().clone() for module in dot.modules]
    layers = metadata["training"]["layers"]
    cohorts = {"early": layers[:2], "late": layers[2:]} if layer_halves else {"all": layers}
    for mode, active in cohorts.items():
        calibration = {}

        def sample(scale):
            with torch.no_grad():
                for layer, module, weight in zip(layers, dot.modules, original, strict=True):
                    module.up.weight.copy_(weight * (scale if layer in active else 0.0))
            calibration[scale] = validation(model, dot, data, offsets)
            return calibration[scale]["forward_kl"]

        selected = largest_scale_under_cap(sample, target)
        assert selected == max(x for x, value in calibration.items() if value["forward_kl"] <= target)
        actual = sample(selected)
        assert actual <= target
        ident = key(seed, recipe, "dot")
        tag = f"-{mode}" if layer_halves else ""
        output = root / f"low-rate-matched-gsm8k{tag}-{seed}.json"
        enriched = {**metadata, "low_rate_kam_adapter_sha256": kam_metadata["adapter_sha256"],
                    "validation_kam_target": target, "dot_validation_kl": actual, "dot_output_scale": selected,
                    "dot_active_layers": active,
                    "validation_grid": [{"scale": x, **record} for x, record in sorted(calibration.items())]}
        score_task(root / "data", output, "gsm8k", 64, model, tokenizer, dot, enriched)
        print(ident, mode, "matched scale", selected, "KAM validation KL", target, "dot validation KL", actual,
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int, choices=range(len(SEEDS)))
    parser.add_argument("--layer-halves", action="store_true")
    args = parser.parse_args()
    measure(args.root, args.index, args.layer_halves)
