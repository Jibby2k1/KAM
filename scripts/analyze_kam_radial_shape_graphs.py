#!/usr/bin/env python3
"""Audit shuffled mixed-length graph replay across GPU types."""

import argparse
import json
from pathlib import Path
from statistics import median

METHODS = ("eager", "shape_graph")


def read(args):
    return sorted((json.loads(f.read_text()) for f in args), key=lambda x: x["seed"])


def summarize(rows):
    good = {}
    for trial in rows:
        checks = trial["comparisons"]
        assert set(checks) == set(trial["runs"]) == {*METHODS, *(m + "_repeat" for m in METHODS)}
        assert all(r["weights"]["equal"] and r["moments"]["equal"] and r["final_loss_equal"] and r["heldout_equal"]
                   for r in checks.values())
        measurements = {}
        for mode in METHODS:
            runs = [trial["runs"][arm] for arm in (mode, mode + "_repeat")]
            measurements[mode] = {k: median(r[k] for r in runs) for k in ("setup_seconds", "wall_seconds", "stream_seconds",
                                                                       "peak_extra_allocated_bytes", "peak_extra_reserved_bytes")}
            measurements[mode]["setup_plus_wall_seconds"] = median(r["setup_seconds"] + r["wall_seconds"] for r in runs)
        eager, graph = [measurements[m] for m in METHODS]
        updates = 3 * trial["cycles"]
        savings = (eager["wall_seconds"] - graph["wall_seconds"]) / updates
        setup = graph["setup_seconds"] - eager["setup_seconds"]
        good[str(trial["seed"])] = {"measurements": measurements, "throughput_gain": eager["wall_seconds"] / graph["wall_seconds"],
                                   "additional_setup_seconds": setup, "simple_amortization_updates": None if savings <= 0
                                   else max(0, setup / savings)}
    return {"device": rows[0]["device"], "updates": 3 * rows[0]["cycles"], "seeds": good,
            "median_times": {m: {key: median(s["measurements"][m][key] for s in good.values())
                                 for key in ("wall_seconds", "setup_plus_wall_seconds", "peak_extra_allocated_bytes")}
                             for m in METHODS}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for m in ("l4", "b200"):
        p.add_argument("--" + m, nargs=3, required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    l4, b200 = read(a.l4), read(a.b200)
    fields = ("seed", "checkpoint_sha256", "cycles", "schedule_sha256", "banks")
    assert all(all(first[k] == second[k] for k in fields) for first, second in zip(l4, b200, strict=True))
    assert all(r["device"] == model for rows, model in ((l4, "NVIDIA L4"), (b200, "NVIDIA B200")) for r in rows)
    data = {"l4": summarize(l4), "b200": summarize(b200),
            "checkpoint_sha256": l4[0]["checkpoint_sha256"], "schedule_sha256": [p["schedule_sha256"] for p in l4]}
    a.output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
