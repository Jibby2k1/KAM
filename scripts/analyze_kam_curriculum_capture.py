#!/usr/bin/env python3
"""Audit exact training equivalence and matched full-curriculum GPU timings."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

from analyze_kam_native_musique import compare

ARCHITECTURES = ("dot", "radial")
MODES = ("eager", "shape_graph")
DEPTHS = ("20", "128", "512", "1024")


def assess(rows):
    out = {}
    for geometry in ARCHITECTURES:
        checks, timings, accuracy = [], [], []
        for trial in rows:
            t = trial["models"][geometry]
            a, b = (t["runs"][m] for m in MODES)
            parity = t["parity"]
            assert parity["weights"]["equal"] and parity["moments"]["equal"]
            assert parity["final_loss_equal"] and parity["heldout_equal"] and a["heldout"] == b["heldout"]
            record = {mode: run | {"incl_setup_seconds": run["setup_seconds"] + run["wall_seconds"]}
                      for mode, run in t["runs"].items()}
            timings.append(record)
            accuracy.append({d: mean(all(z) for z in a["heldout"][d]["outcomes"]) for d in DEPTHS})
            checks.append(a["wall_seconds"] / b["wall_seconds"])
        out[geometry] = {"times": {m: {f: median(r[m][f] for r in timings) for f in (
            "wall_seconds", "incl_setup_seconds", "setup_seconds", "peak_extra_allocated_bytes", "peak_extra_reserved_bytes")}
                                     for m in MODES},
                         "speedup_by_seed": checks,
                         "accuracy": {d: mean(a[d] for a in accuracy) for d in DEPTHS}}
    paired = {}
    for d in DEPTHS:
        effects = []
        for trial in rows:
            a, b = (trial["models"][g]["runs"]["eager"]["heldout"][d] for g in ("radial", "dot"))
            assert a["fixture_sha256"] == b["fixture_sha256"]
            x, y = [[all(c) for c in n["outcomes"]] for n in (a, b)]
            effects.append(compare(x, y, [[i] for i in range(len(x))]))
        paired[d] = effects
    return {"device": rows[0]["device"], "arms": out, "radial_vs_dot_by_seed": paired}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("l4", "b200"):
        p.add_argument("--" + name, type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    p1, p2 = [sorted((json.loads(f.read_text()) for f in s), key=lambda x: x["seed"]) for s in (a.l4, a.b200)]
    fields = ("seed", "steps", "banks", "order_sha256")
    assert [r["device"] for r in p1] == ["NVIDIA L4"]*3 and [r["device"] for r in p2] == ["NVIDIA B200"]*3
    assert all(all(x[k] == y[k] for k in fields) for x, y in zip(p1, p2, strict=True))
    data = {"updates": p1[0]["steps"], "seeds": [x["seed"] for x in p1],
            "cross_device_rate_bytes_equal": [x["rates_sha256"] == y["rates_sha256"] for x, y in zip(p1, p2, strict=True)],
            "l4": assess(p1), "b200": assess(p2)}
    a.output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
