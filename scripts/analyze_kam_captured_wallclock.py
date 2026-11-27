#!/usr/bin/env python3
"""Compare captured training at elapsed checkpoints and audit equal-step replay."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from analyze_kam_native_musique import compare

DEPTHS = ("20", "128", "512", "1024")
GEOMETRIES = ("dot", "radial")


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def compare_device(paths, controls):
    measured = sorted(map(read, paths), key=lambda row: row["seed"])
    baseline = {r["seed"]: r for r in map(read, controls)}
    device, budgets = measured[0]["device"], measured[0]["budgets_seconds"]
    assert len(baseline) == len({r["seed"] for r in measured}) == 3
    panels = {**{f"time/{w}": w for w in budgets}, "step/9600": None, "step/19200": None}
    traces = {}
    for record in measured:
        seed = record["seed"]
        assert record["device"] == device == baseline[seed]["device"]
        assert record["budgets_seconds"] == budgets and record["poll_every_updates"] == 48
        assert set(record["models"]) == set(GEOMETRIES)
        assert {r["initial_shared_sha256"] for r in record["models"].values()} == {
            baseline[seed]["models"]["dot"]["initial_shared_sha256"]}
        for geometry in GEOMETRIES:
            old = baseline[seed]["models"][geometry]["runs"]["shape_graph"]
            new = record["models"][geometry]["checkpoints"]["step/19200"]
            assert new["holdout"] == old["heldout"] and new["loss"] == old["loss"]
        results = {}
        for key, budget in panels.items():
            rows = [record["models"][g]["checkpoints"][key] for g in GEOMETRIES]
            if budget is None:
                assert all(r["step"] == int(key.split("/")[1]) for r in rows)
                assert rows[0]["depth_updates"] == rows[1]["depth_updates"] and rows[0]["tokens"] == rows[1]["tokens"]
            else:
                assert all(r["seconds_with_setup"] >= budget for r in rows)
            assert all(r["step"] % 48 == 0 and sum(r["depth_updates"].values()) == r["step"] for r in rows)
            quality = {}
            for depth in DEPTHS:
                dot, radial = [r["holdout"][depth] for r in rows]
                assert dot["fixture_sha256"] == radial["fixture_sha256"]
                outcomes = [[all(x) for x in track["outcomes"]] for track in (dot, radial)]
                n = len(outcomes[0])
                quality[depth] = {"examples": n, "counts": dict(zip(GEOMETRIES, map(sum, outcomes), strict=True)),
                                  "paired_radial_minus_dot": compare(outcomes[1], outcomes[0], [[i] for i in range(n)])}
            results[key] = {"accuracy": quality, "arms": {
                g: {k: v for k, v in r.items() if k != "holdout"}
                for g, r in zip(GEOMETRIES, rows, strict=True)}}
        traces[str(seed)] = results
    medians = {}
    for key in panels:
        medians[key] = {}
        for g in GEOMETRIES:
            rows = [r[key] for r in traces.values()]
            medians[key][g] = {k: median(r["arms"][g][k] for r in rows)
                               for k in ("step", "tokens", "seconds_with_setup")}
            medians[key][g]["depth_1024_joint"] = median(r["accuracy"]["1024"]["counts"][g] /
                                                        r["accuracy"]["1024"]["examples"] for r in rows)
    return {"device": device, "time_budgets_seconds": budgets, "seeds": traces, "medians": medians,
            "verified_equal_step_forks": len(traces)*len(GEOMETRIES)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("l4", "l4-control", "b200", "b200-control"):
        p.add_argument("--" + name, type=Path, nargs=3, required=name.startswith("l4"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    results = {"l4": compare_device(a.l4, a.l4_control)}
    if a.b200:
        assert a.b200_control
        results["b200"] = compare_device(a.b200, a.b200_control)
    a.output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
