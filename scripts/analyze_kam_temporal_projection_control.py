#!/usr/bin/env python3
"""Audit temporal nonlinear and capacity-matched linear Q/K comparisons."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from analyze_kam_clock_archive_controls import estimate
from analyze_kam_native_musique import compare

POLICIES = ("dot", "spherical", "spherical_linear", "spherical_nonlinear")


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def clean_cases(runs, arm, dataset, depth):
    result = {}
    for row in runs:
        for panel in row["arms"][arm]["archives"][dataset]:
            if panel["depth"] != depth:
                continue
            for r in panel["outcomes"]:
                key = row["seed"], r["id"], r["period"], panel["recode"]
                assert key not in result
                result[key] = r
    return result


def assess(runs):
    assert len(runs) == len({r["seed"] for r in runs}) == 3
    assert {r["device"] for r in runs} == {"NVIDIA L4"}
    assert all(r["zero_residual_parity"] and r["steps"] == 3200 and r["tokens"] == 10556800 for r in runs)
    assert all(r["sources"] == runs[0]["sources"] and r["bottleneck"] == 16 for r in runs)
    param_counts = {k: [r["arms"][k]["parameters"] for r in runs] for k in POLICIES}
    assert param_counts["dot"] == param_counts["spherical"]
    assert param_counts["spherical_linear"] == param_counts["spherical_nonlinear"]
    report = {}
    for policy in POLICIES:
        measurements = {"parameters": param_counts[policy][0],
                        "update_s_median": median(r["arms"][policy]["train"]["seconds"] for r in runs),
                        "seed_seconds": {r["seed"]: r["arms"][policy]["train"]["seconds"] for r in runs}}
        controls = ["dot", "spherical"]
        if policy == "spherical_nonlinear":
            controls.append("spherical_linear")
        natural = {}
        for topic in runs[0]["sources"]:
            natural[topic] = {}
            for d in (16, 48):
                actual = clean_cases(runs, policy, topic, d)
                summary = {"n": len(actual), "strict": sum(v["joint"] for v in actual.values()),
                           "numeric_exact": sum(v["numeric_exact"] for v in actual.values())}
                for control in controls:
                    old = clean_cases(runs, control, topic, d)
                    assert old.keys() == actual.keys()
                    summary[control] = estimate(old, actual, "joint", months=topic != "dallas")
                natural[topic][d] = summary
        synthetic = {}
        for cohort in ("original", "corrections"):
            synthetic[cohort] = {}
            for d in ("16", "48", "128", "512"):
                seed_rows = {}
                for run in runs:
                    found = run["arms"][policy]["heldout"][cohort][d]
                    actual = [all(v) for v in found["outcomes"]]
                    assert len(actual) == 512
                    row = {"n": len(actual), "strict": sum(actual)}
                    for control in controls:
                        old = run["arms"][control]["heldout"][cohort][d]
                        assert old["fixture_sha256"] == found["fixture_sha256"]
                        row[control] = compare(actual, [all(v) for v in old["outcomes"]], [[i] for i in range(len(actual))])
                    seed_rows[run["seed"]] = row
                synthetic[cohort][d] = seed_rows
        report[policy] = measurements | {"archives": natural, "synthetic": synthetic}
    return {"inputs": runs[0]["sources"], "policies": report, "steps": runs[0]["steps"], "train_tokens": runs[0]["tokens"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = sorted(map(read, a.measurements), key=lambda z: z["seed"])
    a.output.write_text(json.dumps(assess(data), sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
