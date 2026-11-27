#!/usr/bin/env python3
"""Compare post-fit dot shrinkage with the original low-rate GSM8K arms."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, key
from summarize_kam_adapter_preservation import BASE, difference, read_archive


def summarize(inputs, archive, uniform=None):
    reference = read_archive(archive)
    results, original, matched, kam, uniform_flags = {}, [], [], [], []
    required = {"source_sha256", "context_tokens", "gpu", "model", "revision", "model_fingerprints"}
    base = reference(f"gsm8k-{BASE}")
    for seed, p in zip(SEEDS, inputs, strict=True):
        calibrated = json.loads(p.read_text())
        meta = calibrated["metadata"]
        dot_reference = reference(f"gsm8k-{key(seed, 'lower_lr', 'dot')}")
        kam_reference = reference(f"gsm8k-{key(seed, 'lower_lr', 'kam')}")
        assert (meta["seed"], meta["recipe"], meta["arm"]) == (seed, "lower_lr", "dot")
        assert all(meta[field] == base["metadata"][field] for field in required)
        assert meta["adapter_sha256"] == dot_reference["metadata"]["adapter_sha256"]
        assert meta["low_rate_kam_adapter_sha256"] == kam_reference["metadata"]["adapter_sha256"]
        controls = {row["scale"]: row["forward_kl"] for row in meta["validation_grid"]}
        scale, cap = meta["dot_output_scale"], meta["validation_kam_target"]
        assert meta["dot_validation_kl"] == controls[scale] <= cap
        assert scale == max(x for x, kl in controls.items() if kl <= cap)
        rows = calibrated["records"]
        assert len(rows) == 1319 and list(range(1319)) == [r["index"] for r in rows]
        assert [r["prompt_tokens"] for r in rows] == [r["prompt_tokens"] for r in base["records"]]
        active = meta.get("dot_active_layers", meta["training"]["layers"])
        if uniform:
            fixed = json.loads(uniform[seed].read_text())
            assert fixed["metadata"]["adapter_sha256"] == meta["adapter_sha256"]
            assert fixed["metadata"]["low_rate_kam_adapter_sha256"] == meta["low_rate_kam_adapter_sha256"]
            assert fixed["metadata"]["validation_kam_target"] == cap
            assert len(active) == 2 and (active == meta["training"]["layers"][:2] or
                                          active == meta["training"]["layers"][2:])
            assert [r["index"] for r in rows] == [r["index"] for r in fixed["records"]]
            uniform_flags.append(np.asarray([r["correct"] for r in fixed["records"]]))
        groups = [np.asarray([r["correct"] for r in group])
                  for group in (rows, dot_reference["records"], kam_reference["records"])]
        current, before, paired = groups
        matched.append(current)
        original.append(before)
        kam.append(paired)
        results[str(seed)] = {"dot_residual_scale": scale, "validation_kam_target": cap,
                              "validation_dot_unscaled": controls[1.0], "validation_dot_matched": controls[scale],
                              "dot_matched_correct": int(current.sum()), "dot_unscaled_correct": int(before.sum()),
                              "kam_native_correct": int(paired.sum()),
                              "dot_matched_parse_count": sum(r["parsed"] for r in rows)}
        if uniform:
            results[str(seed)]["active_layers"] = active
            results[str(seed)]["uniform_matched_correct"] = int(uniform_flags[-1].sum())
    paired_base = np.asarray([r["correct"] for r in base["records"]])[None, :]
    after = np.stack(matched)
    contrasts = {"matched-minus-unscaled": difference(after, np.stack(original)),
                 "matched-minus-kam": difference(after, np.stack(kam)),
                 "matched-minus-untouched": difference(after, paired_base)}
    if uniform:
        contrasts["matched-minus-uniform"] = difference(after, np.stack(uniform_flags))
    return {"model": base["metadata"]["model"], "n": 1319, "seeds": SEEDS, "seed_details": results,
            "contrast": contrasts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs=len(SEEDS), type=Path)
    parser.add_argument("--uniform", nargs=len(SEEDS), type=Path)
    args = parser.parse_args()
    uniform = dict(zip(SEEDS, args.uniform, strict=True)) if args.uniform else None
    results = summarize(args.inputs, args.archive, uniform)
    args.output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
