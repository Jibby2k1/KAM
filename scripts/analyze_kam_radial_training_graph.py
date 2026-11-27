#!/usr/bin/env python3
"""Verify same-policy capture parity and summarize matched update timings."""

import argparse
import json
from pathlib import Path
from statistics import median

NAMES = ("eager_fp32", "fp32_short_flash_long")


def unpack(path):
    return json.loads(path.read_text())


def assess(records, *, cursor, deterministic):
    records = sorted(records, key=lambda row: row["seed"])
    assert len({r["checkpoint_sha256"] for r in records}) == 1
    assert len({r["cycles"] for r in records}) == len({r["update_tokens"] for r in records}) == 1
    assert all(r["deterministic_algorithms"] is deterministic for r in records)
    results = {}
    for name in NAMES:
        timings, comparisons, setup = {}, [], {}
        modes = ("regular", "graph", "cursor") if cursor else ("regular", "graph")
        for row in records:
            arm = row["policies"][name]
            base = arm["runs"]["regular"]
            assert all(path["held_out"][key]["fixture_sha256"] == base["held_out"][key]["fixture_sha256"]
                       for path in arm["runs"].values() for key in base["held_out"])
            assert set(arm["runs"]) == set(modes) | {f"{m}_repeat" for m in modes}
            if deterministic:
                assert arm["weight_difference"]["equal"] and all(d["equal"] for d in arm["moment_difference"].values())
                assert all(d["weights"]["equal"] and d["moments"]["equal"] for d in arm["same_path_repeat"].values())
                if cursor:
                    assert all(d["equal"] for d in arm["cursor_difference"].values())
                assert all(run["held_out"] == base["held_out"] and run["final_losses"] == base["final_losses"]
                           for run in arm["runs"].values())
            timings[row["seed"]] = {mode: median([arm["runs"][mode]["seconds"], arm["runs"][f"{mode}_repeat"]["seconds"]])
                                    for mode in modes}
            setup[row["seed"]] = {mode: median([arm["runs"][mode]["setup_seconds"],
                                               arm["runs"][f"{mode}_repeat"]["setup_seconds"]]) for mode in modes}
            comparisons.append({"seed": row["seed"],
                                "graph_weight_l2": arm["weight_difference"]["relative_l2"],
                                "eager_repeat_weight_l2": arm["same_path_repeat"]["regular"]["weights"]["relative_l2"],
                                "graph_repeat_weight_l2": arm["same_path_repeat"]["graph"]["weights"]["relative_l2"],
                                "all_held_out_equal": all(run["held_out"] == base["held_out"] for run in arm["runs"].values())})
        results[name] = {"timing_by_seed": timings, "setup_by_seed": setup,
                         "median_seconds": {mode: median([v[mode] for v in timings.values()]) for mode in modes},
                         "paired_speedup_vs_eager": {mode: [v["regular"] / v[mode] for v in timings.values()]
                                                     for mode in modes[1:]},
                         "comparisons": comparisons}
    return {"device": records[0]["device"], "checkpoint_sha256": records[0]["checkpoint_sha256"],
            "cycles": records[0]["cycles"], "update_tokens": records[0]["update_tokens"], "policies": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l4-constant", nargs=3, type=Path, required=True)
    p.add_argument("--l4-cosine", nargs=3, type=Path, required=True)
    p.add_argument("--b200-constant", nargs=3, type=Path)
    p.add_argument("--l4-nondeterministic", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    scores = {"l4_constant": assess(list(map(unpack, a.l4_constant)), cursor=False, deterministic=True),
              "l4_cosine": assess(list(map(unpack, a.l4_cosine)), cursor=True, deterministic=True),
              "l4_nondeterministic": assess([unpack(a.l4_nondeterministic)], cursor=False, deterministic=False)}
    if a.b200_constant:
        scores["b200_constant"] = assess(list(map(unpack, a.b200_constant)), cursor=False, deterministic=True)
    a.output.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
