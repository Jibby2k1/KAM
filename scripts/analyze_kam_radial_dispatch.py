#!/usr/bin/env python3
"""Audit matched quality, memory and throughput for the depth-selective radial kernel."""

import argparse
import json
from pathlib import Path
from statistics import mean

from analyze_kam_radial_backward import CHECKPOINT, DEPTHS
from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

SEEDS = (638982, 639003, 639051)
MODES = ("eager_fp32", "eager_bf16", "augmented_flash_bf16", "balanced_split_flash_bf16", "fp32_short_flash_long")
SCHEDULE = [[20, 100, 4096], [128, 16, 2048], [512, 4, 1024]]
VIEWS = ("common_fp32", "native")


def measure(raw):
    assert len(raw) == 512 and all(len(t) == 2 and all(isinstance(x, bool) for x in t) for t in raw)
    return {"address": sum(x[0] for x in raw), "record": sum(x[1] for x in raw), "joint": sum(all(x) for x in raw)}


def inspect(run):
    assert run["checkpoint_sha256"] == CHECKPOINT and run["checkpoint_tokens"] == 13_516_800
    assert run["schedule"] == SCHEDULE
    assert set(run["initial"]) == set(run["banks"]) == set(DEPTHS)
    order = run["order"]
    assert tuple(run["modes"]) == (*MODES[order:], *MODES[:order])
    eager = run["modes"]["eager_fp32"]
    assert eager["native"] == eager["common_fp32"]
    initial = {d: measure(x["outcomes"]) for d, x in run["initial"].items()}
    stamps = {d: r["fixture_sha256"] for d, r in run["initial"].items()}
    metrics = {}
    for mode, result in run["modes"].items():
        fit = result["training"]
        assert fit["steps"] == 1200 and fit["tokens"] == 4_924_800
        assert fit["seconds"] > 0 and fit["peak_extra_allocated_bytes"] > 0
        assert [p["step"] for p in fit["losses"]] == [300, 600, 900, 1200]
        assert all(0 <= p["loss"] < float("inf") and p["tokens"] * 1200 == fit["tokens"] * p["step"]
                   for p in fit["losses"])
        cells = {"training": fit}
        for view in VIEWS:
            assert {d: p["fixture_sha256"] for d, p in result[view].items()} == stamps
            cells[view] = {d: measure(p["outcomes"]) for d, p in result[view].items()}
        cells["paired_joint_vs_eager_fp32"] = {
            d: paired([all(x) for x in result["common_fp32"][d]["outcomes"]],
                      [all(x) for x in eager["common_fp32"][d]["outcomes"]])
            for d in DEPTHS
        }
        metrics[mode] = cells
    identity = (run["checkpoint_sha256"], run["order"], run["banks"], stamps)
    return {"initial_correct_of_512": initial, "modes": metrics}, identity


def summarize(runs):
    output, stamps = {}, {}
    for name, gpu in (("b200", "NVIDIA B200"), ("l4", "NVIDIA L4")):
        jobs = {r["seed"]: r for r in runs[name]}
        assert len(jobs) == len(runs[name]) == len(SEEDS) and set(jobs) == set(SEEDS)
        details = {}
        for position, seed in enumerate(SEEDS):
            assert jobs[seed]["device"] == gpu and jobs[seed]["order"] == position
            details[str(seed)], identity = inspect(jobs[seed])
            assert stamps.setdefault(seed, identity) == identity
        averages = {}
        for mode in MODES:
            observations = [detail["modes"][mode] for detail in details.values()]
            averages[mode] = {
                "training_seconds": round(mean(x["training"]["seconds"] for x in observations), 6),
                "peak_extra_bytes": round(mean(x["training"]["peak_extra_allocated_bytes"] for x in observations)),
                **{view: {d: {m: round(mean(o[view][d][m] / 512 for o in observations), 8)
                              for m in ("address", "record", "joint")} for d in DEPTHS} for view in VIEWS},
            }
        output[name] = {"per_seed": details, "mean": averages}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b200", required=True, nargs=3, type=Path)
    p.add_argument("--l4", required=True, nargs=3, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = summarize({k: [unpack(f) for f in getattr(a, k)] for k in ("b200", "l4")})
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
