#!/usr/bin/env python3
"""Compare one intermediate KL weight with frozen low-rate endpoints."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, LOW_RATE_KL_RECIPE, LOW_RATE_KL_RUNS, MODEL_REGISTRY, SEEDS, key
from summarize_kam_adapter_preservation import BASE, difference, read_archive
from summarize_kam_general_benchmarks import TASKS, details


def summarize(pilot_path, fixed_path):
    pilot, fixed = read_archive(pilot_path), read_archive(fixed_path)
    model = fixed(f"mmlu-{BASE}")["metadata"]["model"]
    assert model in MODEL_REGISTRY
    fits = {}
    for seed, recipe, arm in LOW_RATE_KL_RUNS:
        ident = key(seed, recipe, arm)
        fit = pilot(f"train-{ident}")
        match = fixed(f"train-{key(seed, 'lower_lr', arm)}")
        assert (fit["model"], fit["seed"], fit["recipe"], fit["arm"], fit["peak_lr"], fit["kl_weight"]) == (
            model, seed, recipe, arm, 8e-5, .5,
        )
        for field in ("revision", "layers", "train_tokens", "exposure_sha256", "token_manifest_sha256",
                      "model_fingerprints", "new_trainable_parameters", "gpu"):
            assert fit[field] == match[field], (ident, field)
        assert fit["training"][-1]["step"] == match["training"][-1]["step"] == 256
        fits[ident] = fit

    result = {}
    for task, n in TASKS.items():
        codes = task in ("humaneval", "mbpp")
        suffix = "-graded" if codes else ""
        fields = ["source_sha256", "model_fingerprints", "context_tokens", "gpu"]
        if codes:
            fields.append("sandbox_sha256")
        controls = fixed(f"{task}-{BASE}{suffix}")
        base = np.asarray([r["correct"] for r in controls["records"]])
        item_flags, scores = {}, {}
        for recipe in ("lower_lr", LOW_RATE_KL_RECIPE, "lower_lr_kl"):
            read = pilot if recipe == LOW_RATE_KL_RECIPE else fixed
            for arm in ARMS:
                per_seed = []
                for seed in SEEDS:
                    ident = key(seed, recipe, arm)
                    measured = read(f"{task}-{ident}{suffix}")
                    if recipe == LOW_RATE_KL_RECIPE:
                        assert measured["metadata"]["adapter_sha256"] == fits[ident]["adapter_sha256"]
                    assert all(measured["metadata"][k] == controls["metadata"][k] for k in fields)
                    rows = measured["records"]
                    assert len(rows) == n
                    assert [(r["index"], r["prompt_tokens"]) for r in rows] == [
                        (r["index"], r["prompt_tokens"]) for r in controls["records"]
                    ]
                    per_seed.append([r["correct"] for r in rows])
                items = np.asarray(per_seed)
                item_flags[recipe, arm] = items
                scores[f"{recipe}-{arm}"] = {"mean": float(items.mean()),
                                             "correct_by_seed": items.sum(axis=1).tolist(),
                                             "vs_untouched": difference(items, base[None, :])}
        effects = {f"kl05_minus_{r}:{a}": difference(item_flags[LOW_RATE_KL_RECIPE, a], item_flags[r, a])
                   for r in ("lower_lr", "lower_lr_kl") for a in ARMS}
        effects["kl05:kam_minus_dot"] = difference(item_flags[LOW_RATE_KL_RECIPE, "kam"],
                                                   item_flags[LOW_RATE_KL_RECIPE, "dot"])
        result[task] = {"n": n, "untouched": details(controls["records"]),
                        "seed_means": scores, "contrasts": effects}
    return {"model": model, "seeds": SEEDS, "training": fits, "tasks": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("factorial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.pilot, args.factorial)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
