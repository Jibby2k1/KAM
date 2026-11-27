#!/usr/bin/env python3
"""Summarize matched wall-clock geometry checkpoints with paired outcomes."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

from analyze_kam_native_musique import compare

MODES = ("dot_eager", "dot_selective", "radial_selective")
BUDGETS = ("25.0", "45.0", "60.0")
DEPTHS = ("20", "128", "512", "1024")


def holdout(checkpoint, depth):
    p = checkpoint["holdout"][depth]
    return p["fixture_sha256"], [all(row) for row in p["outcomes"]]


def compare_depths(rows, mode, baseline, checkpoint, depth):
    differences = []
    for record in rows:
        mine, control = [record["models"][k]["checkpoints"][checkpoint] for k in (mode, baseline)]
        hash_m, test = holdout(mine, depth)
        hash_c, other = holdout(control, depth)
        assert hash_m == hash_c
        differences.append(compare(test, other, [[i] for i in range(len(test))]))
    return differences


def inspect(rows, checkpoint):
    blocks = {mode: [r["models"][mode]["checkpoints"][checkpoint] for r in rows] for mode in MODES}
    output = {}
    for mode, trials in blocks.items():
        if checkpoint in BUDGETS:
            assert all(t["seconds"] >= float(checkpoint) for t in trials)
        output[mode] = {"median_seconds": median(r["seconds"] for r in trials),
                        "seconds": [r["seconds"] for r in trials],
                        "updates": [r["step"] for r in trials],
                        "tokens": [r["tokens"] for r in trials],
                        "peak_extra_allocated_bytes": [r["peak_extra_allocated_bytes"] for r in trials],
                        "depth_updates": [r["depth_updates"] for r in trials],
                        "mean_joint": {d: mean(mean(holdout(r, d)[1]) for r in trials) for d in DEPTHS}}
    comparisons = (("dot_selective", "dot_eager"), ("radial_selective", "dot_selective"), ("radial_selective", "dot_eager"))
    by_depth = {f"{mode}_vs_{base}": {d: compare_depths(rows, mode, base, checkpoint, d) for d in DEPTHS}
                for mode, base in comparisons}
    return {"modes": output, "pairs_by_seed": by_depth}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", nargs=3, required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    rows = sorted((json.loads(p.read_text()) for p in args.measurements), key=lambda r: r["seed"])
    assert len({r["seed"] for r in rows}) == 3 and all(r["device"] == "NVIDIA L4" for r in rows)
    assert all(r["shape"] == rows[0]["shape"] and [str(b) for b in r["budgets"]] == list(BUDGETS) for r in rows)
    for run in rows:
        models = list(run["models"].values())
        assert set(run["models"]) == set(MODES)
        assert len({m["initial_shared_sha256"] for m in models}) == 1
        for model in models:
            for k in BUDGETS:
                assert model["checkpoints"][k]["step"] % 120 == 0
    data = {"seeds": [r["seed"] for r in rows], "device": rows[0]["device"],
            "dot_precision_check_by_seed": [r["dot_precision_probe"]["checks"] for r in rows],
            "budgets": {k: inspect(rows, k) for k in BUDGETS},
            "step_4800_control": inspect(rows, "step/4800")}
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
