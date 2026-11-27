#!/usr/bin/env python3
"""Audit the matched radial-attention training runs and summarize their results."""

import argparse
import json
from math import isfinite
from pathlib import Path

from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

DEPTHS = ("20", "128", "512")
BATCHES = ("1", "16", "32")
MODES = {"eager_fp32", "augmented_math_fp32", "eager_bf16", "augmented_flash_bf16"}
CHECKPOINT = "3e1196f8f88c42e812a05580a52438f167135ff9129e5086af59c14d6ffd777f"


def fixture_identity(run):
    return (
        run["checkpoint_sha256"], run["checkpoint_train_tokens"], run["seed"],
        run["timing_repeats"], run["continuation_input_sha256"],
        tuple(run["gradients"][d]["input_sha256"] for d in DEPTHS),
        tuple(run["held_out"][d]["fixture_sha256"] for d in DEPTHS),
    )


def summarize_run(run):
    assert run["checkpoint_sha256"] == CHECKPOINT
    assert run["timing_repeats"] == 24
    assert set(run["gradients"]) == set(run["training_step"]) == set(run["held_out"]) == set(DEPTHS)
    assert set(run["continuation"]) == MODES
    for row in run["continuation"].values():
        assert row["steps"] == 128 and row["tokens"] == 305_152 and row["elapsed_seconds"] > 0
        assert [point["step"] for point in row["loss"]] == [32, 64, 96, 128]
        assert all(isfinite(point["loss"]) for point in row["loss"])

    metrics = {}
    for depth in DEPTHS:
        grad = run["gradients"][depth]
        assert set(grad) == MODES | {"input_sha256"}
        math = grad["augmented_math_fp32"]
        flash = grad["augmented_flash_bf16"]
        assert math["reference"] == "eager_fp32" and flash["reference"] == "eager_bf16"
        for result in (math, flash):
            assert set(result["gradient_error"]) == {"all", "metric_and_bandwidth", "qkv"}
            assert all(isfinite(v) for family in result["gradient_error"].values() for v in family.values())

        timing = {}
        assert set(run["training_step"][depth]) == set(BATCHES)
        for batch in BATCHES:
            options = run["training_step"][depth][batch]
            assert set(options) == MODES
            for row in options.values():
                assert 0 < row["p50_ms"] <= row["p95_ms"] < float("inf")
                assert row["peak_extra_allocated_bytes"] >= 0
                tokens = int(batch) * (2 * int(depth) + 1)
                assert abs(row["tokens_per_second"] * row["p50_ms"] / 1000 - tokens) < 1e-6
            eager, fused = options["eager_bf16"], options["augmented_flash_bf16"]
            timing[batch] = {
                "p50_speedup": round(eager["p50_ms"] / fused["p50_ms"], 6),
                "p95_speedup": round(eager["p95_ms"] / fused["p95_ms"], 6),
                "peak_extra_saved_bytes": eager["peak_extra_allocated_bytes"] - fused["peak_extra_allocated_bytes"],
            }

        held = run["held_out"][depth]
        assert held["examples"] == 512 and set(held["outcomes"]) == MODES
        for row in held["outcomes"].values():
            assert set(row) == {"correct", "agrees_with_fp32"}
            assert all(len(values) == 512 and set(values) <= {0, 1} for values in row.values())
        eager_fp, eager_bf = (held["outcomes"][mode]["correct"] for mode in ("eager_fp32", "eager_bf16"))
        assert all(held["outcomes"]["eager_fp32"]["agrees_with_fp32"])
        metrics[depth] = {
            "fp32_tolerances_met": math["fp32_logit_tolerance"] and math["gradient_error"]["all"]["relative_l2"] <= 1e-3,
            "flash_bf16_vs_eager_bf16_timing": timing,
            "correct": {mode: sum(row["correct"]) for mode, row in held["outcomes"].items()},
            "agrees_with_fp32": {mode: sum(row["agrees_with_fp32"]) for mode, row in held["outcomes"].items()},
            "paired_accuracy": {
                "augmented_math_fp32_vs_eager_fp32": paired(held["outcomes"]["augmented_math_fp32"]["correct"], eager_fp),
                "augmented_flash_bf16_vs_eager_bf16": paired(held["outcomes"]["augmented_flash_bf16"]["correct"], eager_bf),
            },
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b200", type=Path, required=True)
    parser.add_argument("--l4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {"b200": unpack(args.b200), "l4": unpack(args.l4)}
    assert "B200" in runs["b200"]["device"] and runs["l4"]["device"].endswith("L4")
    assert fixture_identity(runs["b200"]) == fixture_identity(runs["l4"])
    summary = {key: {"device": row["device"], "torch": row["torch"], "depths": summarize_run(row)}
               for key, row in runs.items()}
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
