#!/usr/bin/env python3
"""Audit curriculum order under shared CUDA Graphs and elapsed budgets."""

import argparse
import json
from pathlib import Path
from statistics import median

from analyze_kam_captured_wallclock import DEPTHS, GEOMETRIES, read
from analyze_kam_native_musique import compare

POLICIES = ("short_first", "interleaved")


def analyze(paths, references):
    runs = sorted(map(read, paths), key=lambda p: p["seed"])
    known = {r["seed"]: r for r in map(read, references)}
    assert {r["seed"] for r in runs} == known.keys()
    budgets = runs[0]["budgets_seconds"]
    keys = [f"time/{j}" for j in budgets] + [f"step/{j}" for j in (4800, 9600, 19200)]
    devices = {r["device"] for r in runs} | {r["device"] for r in known.values()}
    assert len(devices) == 1
    profiles, exact = {}, 0
    for r in runs:
        seed = r["seed"]
        prior = known[seed]
        assert all(r[f] == prior[f] for f in ("order_sha256", "rates_sha256", "banks", "budgets_seconds"))
        assert r["order_stamps"]["short_first"] == r["order_sha256"]
        assert r["order_stamps"]["short_first"] != r["order_stamps"]["interleaved"]
        labels = {}
        for arm in GEOMETRIES:
            rows = [r["models"][f"{arm}/{p}"] for p in POLICIES]
            assert all(m["initial_shared_sha256"] == prior["models"][arm]["initial_shared_sha256"] for m in rows)
            for point in ("step/9600", "step/19200"):
                before = prior["models"][arm]["checkpoints"][point]
                after = rows[0]["checkpoints"][point]
                assert all(before[f] == after[f] for f in ("step", "tokens", "depth_updates", "loss", "holdout"))
                exact += 1
            labels[arm] = {}
            for point in keys:
                sample = [row["checkpoints"][point] for row in rows]
                if point in ("step/9600", "step/19200"):
                    assert all(sample[0][k] == sample[1][k] for k in ("depth_updates", "tokens"))
                measure = {}
                for depth in DEPTHS:
                    before, after = [r["holdout"][depth] for r in sample]
                    assert before["fixture_sha256"] == after["fixture_sha256"]
                    x, y = [list(map(all, v["outcomes"])) for v in (before, after)]
                    measure[depth] = {"tested": len(x), "short_first": sum(x), "interleaved": sum(y),
                                      "short_vs_interleaved": compare(x, y, [[i] for i in range(len(x))])}
                labels[arm][point] = {"heldout": measure, "arms": {
                    m: {k: v for k, v in values.items() if k != "holdout"}
                    for m, values in zip(POLICIES, sample, strict=True)}}
        profiles[str(seed)] = labels
    median_result = {}
    for arm in GEOMETRIES:
        median_result[arm] = {}
        for point in keys:
            rows = [r[arm][point] for r in profiles.values()]
            median_result[arm][point] = {mode: {
                "updates": median(r["arms"][mode]["step"] for r in rows),
                "time_with_setup": median(r["arms"][mode]["seconds_with_setup"] for r in rows),
                "depth_1024_joint": median(r["heldout"]["1024"][mode] / r["heldout"]["1024"]["tested"] for r in rows)}
                for mode in POLICIES}
    return {"device": devices.pop(), "exact_prior_checkpoints": exact, "medians": median_result, "seeds": profiles}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("l4", "l4-control", "b200", "b200-control"):
        p.add_argument("--" + name, required=name.startswith("l4"), nargs=3, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = {"l4": analyze(a.l4, a.l4_control)}
    if a.b200:
        assert a.b200_control
        result["b200"] = analyze(a.b200, a.b200_control)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
