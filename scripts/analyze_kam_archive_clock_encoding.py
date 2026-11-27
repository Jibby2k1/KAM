#!/usr/bin/env python3
"""Compare archive-shaped, corrected, and mixed training on shared clock probes."""

import argparse
from pathlib import Path
import json
from statistics import mean

from analyze_kam_temporal_clock_encoding import GEOMETRIES, read_measurement, summarize

MODES = ("linear", "signed_log", "signed_log_guard", "signed_log_compact")
PAIRS = tuple((mode, "linear") for mode in MODES[1:])


def split_inputs(paths, domain):
    rows = {r["seed"]: r for r in map(read_measurement, paths)}
    assert len(rows) == 3
    assert all(r.get("training_domain", "corrections") == domain and r["device"] == "NVIDIA L4"
               for r in rows.values())
    return rows


def check(reference, observed):
    assert reference.keys() == observed.keys()
    for seed, old in reference.items():
        new = observed[seed]
        assert all(old[k] == new[k] for k in ("steps", "schedule", "archive_sha256"))
        assert all(new["banks"][d]["sample_sha256"] == r["sample_sha256"] for d, r in old["banks"].items())
        for name, shape in old["results"].items():
            fresh = new["results"][name]
            assert shape["initial_sha256"] == fresh["initial_sha256"]
            for domain, depths in shape["procedural"].items():
                assert all(v["fixture_sha256"] == fresh["procedural"][domain][d]["fixture_sha256"]
                           for d, v in depths.items())
            for era, panels in shape["archive"].items():
                for before, after in zip(panels, fresh["archive"][era], strict=True):
                    assert before["fixture_sha256"] == after["fixture_sha256"]


def header(seeds):
    by_condition = {}
    for geometry in GEOMETRIES:
        aggregates = {}
        for name in ("original", "corrections", "archive"):
            if name == "archive":
                panels = {f"{era}/{depth}": [q[geometry]["archive"][era][depth]["joint"] for q in seeds.values()]
                          for era in ("earlier", "later") for depth in ("16", "48")}
            else:
                panels = {depth: [q[geometry]["procedures"][name][depth] for q in seeds.values()]
                          for depth in ("16", "48", "128", "512")}
            aggregates[name] = {k: {m: mean(r["counts"][m] / r["n"] for r in v) for m in MODES} for k, v in panels.items()}
        by_condition[geometry] = aggregates
    return by_condition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("archive", "corrections", "compaction"):
        p.add_argument("--" + option, type=Path, nargs=3, required=True)
    p.add_argument("--mixed", type=Path, nargs=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prior, compact = (split_inputs(m, "corrections") for m in (a.corrections, a.compaction))
    assert prior.keys() == compact.keys()
    reference = {}
    for seed in sorted(prior):
        old, new = prior[seed], compact[seed]
        assert all(old[k] == new[k] for k in ("banks", "schedule", "steps", "archive_sha256"))
        for name, record in new["results"].items():
            if name in old["results"]:
                assert all(record[k] == old["results"][name][k] for k in ("initial_sha256", "procedural", "archive"))
        reference[seed] = old | {"results": old["results"] | new["results"]}
    found = {"corrections": reference, "archive_shaped": split_inputs(a.archive, "archive_shaped")}
    if a.mixed:
        found["mixed"] = split_inputs(a.mixed, "mixed")
    conditions = {}
    for domain, measurements in found.items():
        check(reference, measurements)
        seeds = {str(seed): summarize(measurements[seed], MODES, PAIRS) for seed in sorted(measurements)}
        conditions[domain] = {"means": header(seeds), "seeds": seeds,
                              "training_banks": {str(seed): m["banks"] for seed, m in measurements.items()}}
    result = {"steps": next(iter(reference.values()))["steps"],
              "archive_sha256": next(iter(reference.values()))["archive_sha256"], "conditions": conditions}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
