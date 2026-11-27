#!/usr/bin/env python3
"""Compare equivalent BF16 radial feature encodings on matched training seeds."""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import mean

from analyze_kam_radial_backward import CHECKPOINT, DEPTHS
from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

BASELINE = "augmented_flash_bf16"
VARIANTS = ("split_flash_bf16", "balanced_flash_bf16", "balanced_split_flash_bf16")
MODES = {"eager_fp32", "eager_bf16", BASELINE, *VARIANTS}
SEEDS = ("829114", "829128", "829162")
SHAPES = ("128/16", "512/16", "512/32")


def fixture_identity(run):
    return (
        run["checkpoint_sha256"],
        tuple(run["timing"][shape]["input_sha256"] for shape in SHAPES),
        tuple((run["seeds"][seed]["updates_sha256"],
               tuple(run["seeds"][seed]["gradients"][d]["input_sha256"] for d in DEPTHS),
               tuple(run["seeds"][seed]["held_out"][d]["input_sha256"] for d in DEPTHS))
              for seed in SEEDS),
    )


def summarize_run(run):
    assert run["checkpoint_sha256"] == CHECKPOINT
    assert set(run["seeds"]) == set(SEEDS)
    assert set(run["timing"]) == set(SHAPES)
    timing, quality = {}, {}
    for shape in SHAPES:
        options = run["timing"][shape]["steps"]
        assert set(options) == MODES
        assert all(0 < row["p50_ms"] <= row["p95_ms"] < float("inf") and
                   row["peak_extra_allocated_bytes"] >= 0 for row in options.values())
        timing[shape] = {mode: row | {
            "p50_speedup_vs_eager_bf16": round(options["eager_bf16"]["p50_ms"] / row["p50_ms"], 6),
            "p50_overhead_vs_original_flash": round(row["p50_ms"] / options[BASELINE]["p50_ms"], 6),
        } for mode, row in options.items()}
    for seed in SEEDS:
        data = run["seeds"][seed]
        assert set(data["gradients"]) == set(data["held_out"]) == set(DEPTHS)
        assert set(data["continuation"]) == MODES
        for row in data["continuation"].values():
            assert row["steps"] == 128 and row["tokens"] == 305_152 and row["elapsed_seconds"] > 0
            assert [p["step"] for p in row["loss"]] == [32, 64, 96, 128]
            assert all(isfinite(p["loss"]) for p in row["loss"])
        depths = {}
        for depth in DEPTHS:
            grads = data["gradients"][depth]["measurements"]
            assert set(grads) == MODES and all(v["reference"] == "eager_fp32" for v in grads.values())
            for row in grads.values():
                assert set(row["gradient_error"]) == {"all", "metric_and_bandwidth", "qkv"}
                assert all(isfinite(v) for family in row["gradient_error"].values() for v in family.values())
            cases = data["held_out"][depth]["outcomes"]
            assert set(cases) == MODES
            for row in cases.values():
                assert set(row) == {"correct", "agrees_with_fp32"}
                assert all(len(v) == 512 and set(v) <= {0, 1} for v in row.values())
            assert all(cases["eager_fp32"]["agrees_with_fp32"])
            depths[depth] = {
                "correct": {m: sum(v["correct"]) for m, v in cases.items()},
                "gradient_relative_l2": {m: v["gradient_error"]["all"]["relative_l2"] for m, v in grads.items()},
                "paired_vs_eager_bf16": {m: paired(cases[m]["correct"], cases["eager_bf16"]["correct"])
                                         for m in (BASELINE, *VARIANTS)},
                "paired_vs_original_flash": {m: paired(cases[m]["correct"], cases[BASELINE]["correct"])
                                              for m in VARIANTS},
            }
        quality[seed] = depths
    averages = {d: {m: round(mean(quality[s][d]["correct"][m] / 512 for s in SEEDS), 8)
                    for m in sorted(MODES)} for d in DEPTHS}
    return {"timing": timing, "per_seed": quality, "mean_accuracy_across_seeds": averages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b200", type=Path, required=True)
    parser.add_argument("--l4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {"b200": unpack(args.b200), "l4": unpack(args.l4)}
    assert "B200" in runs["b200"]["device"] and runs["l4"]["device"].endswith("L4")
    assert fixture_identity(runs["b200"]) == fixture_identity(runs["l4"])
    summary = {key: {"device": row["device"], "torch": row["torch"], **summarize_run(row)}
               for key, row in runs.items()}
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
