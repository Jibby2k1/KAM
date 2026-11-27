#!/usr/bin/env python3
"""Check optimizer-state forks and paired fixed-update learning-rate effects."""

import argparse
import json
from pathlib import Path
from statistics import mean

from analyze_kam_native_musique import compare

ARMS = ("constant", "cosine")
DEPTHS = ("20", "128", "512", "1024")


def summarize(rows, geometry, step):
    quality, change = {a: {d: [] for d in DEPTHS} for a in ARMS}, {d: [] for d in DEPTHS}
    for trial in rows:
        p = trial["models"][geometry]["branches"]
        before, after = [p[a][step] for a in ARMS]
        assert before["depth_updates"] == after["depth_updates"] and sum(before["depth_updates"].values()) == int(step)
        assert before["rate"] > after["rate"]
        for d in DEPTHS:
            panels = [r["held_out"][d] for r in (before, after)]
            assert panels[0]["fixture_sha256"] == panels[1]["fixture_sha256"]
            observed = [[all(choices) for choices in p["outcomes"]] for p in panels]
            for a, outcomes in zip(ARMS, observed, strict=True):
                quality[a][d].append(mean(outcomes))
            change[d].append(compare(observed[1], observed[0], [[i] for i in range(len(observed[0]))]))
    return {"mean_joint": {a: {d: mean(row) for d, row in results.items()} for a, results in quality.items()},
            "cosine_vs_constant_by_seed": change}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("measurements", "wallclock"):
        p.add_argument("--" + name, nargs=3, required=True, type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    trials, earlier = [sorted((json.loads(f.read_text()) for f in files), key=lambda r: r["seed"])
                       for files in (a.measurements, a.wallclock)]
    assert all(r["device"] == "NVIDIA L4" for r in (*trials, *earlier))
    geometries = ("dot", "radial")
    parity = {}
    for new, old in zip(trials, earlier, strict=True):
        assert new["seed"] == old["seed"] and new["bank_sha256"] == old["banks"]
        parity[new["seed"]] = {}
        for g in geometries:
            prefix = new["models"][g]["prefix"]["9600"]["held_out"]
            reference = old["models"][g + "_selective"]["checkpoints"]["step/9600"]["holdout"]
            assert all(prefix[d]["fixture_sha256"] == reference[d]["fixture_sha256"] for d in DEPTHS)
            parity[new["seed"]][g] = prefix == reference
    data = {"seeds": [r["seed"] for r in trials], "prefix_exact_agreement": parity,
            "matched_checkpoints": {s: {g: summarize(trials, g, s) for g in geometries}
                                    for s in ("12000", "14400", "16800", "19200")}}
    a.output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
