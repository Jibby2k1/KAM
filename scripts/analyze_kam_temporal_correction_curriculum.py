#!/usr/bin/env python3
"""Check matched temporal correction curricula and archival transfer."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from analyze_kam_native_musique import compare

GEOMETRIES = ("dot", "radial")
CONDITIONS = ("original", "corrections")


def pairs(rows, labels, identities):
    clusters = defaultdict(list)
    for i, group in enumerate(identities):
        clusters[group].append(i)
    return {condition: sum(row) for condition, row in zip(labels, rows, strict=True)} | {
        "correction_vs_original": compare(rows[1], rows[0], list(clusters.values())), "questions": len(identities)}


def synthetic(models, test, depth):
    outcomes = [m["synthetic"][test][depth] for m in models]
    assert outcomes[0]["fixture_sha256"] == outcomes[1]["fixture_sha256"]
    measures = [[all(entry) for entry in arm["outcomes"]] for arm in outcomes]
    return pairs(measures, CONDITIONS, list(range(len(measures[0]))))


def archive(models, test, depth):
    panels = [[p for p in m["archive"][test] if p["depth"] == depth] for m in models]
    assert [p["fixture_sha256"] for p in panels[0]] == [p["fixture_sha256"] for p in panels[1]]
    flat = [[r for p in items for r in p["outcomes"]] for items in panels]
    names = [[(r["id"], r["period"], r["series"]) for r in arm] for arm in flat]
    assert names[0] == names[1]
    groups = [r["period"] for r in flat[0]]
    return {key: pairs([[r[key] for r in arm] for arm in flat], CONDITIONS, groups) for key in ("joint", "numeric_exact")}


def analyze(record):
    modes = record["outcomes"]
    assert len({m["initial_shared_sha256"] for m in modes.values()}) == 1
    steps = str(record["training_steps"])
    traces = [m["training"][steps] for m in modes.values()]
    assert len({t["tokens"] for t in traces}) == 1 and all(t["depth_updates"] == traces[0]["depth_updates"] for t in traces)
    for d, raw in record["baseline_banks"].items():
        changed = record["shifted_banks"][d]
        assert raw[0] != changed[0] and raw[1] == changed[1]
    output = {}
    for geometry in GEOMETRIES:
        selected = [modes[f"{geometry}/{name}"] for name in CONDITIONS]
        cohorts = {name: {d: synthetic(selected, name, d) for d in ("48", "128", "512", "1024")} for name in CONDITIONS}
        transfers = {name: {d: archive(selected, name, d) for d in (16, 48)} for name in ("earlier", "later")}
        output[geometry] = {"procedural": cohorts, "archive": transfers,
                            "train_seconds": [m["training"][steps]["elapsed_update_seconds"] for m in selected]}
    return {"seed": record["seed"], "counts_changed": record["changed_records"], "arms": output,
            "tokens": traces[0]["tokens"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", nargs=3, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    results = sorted((json.loads(f.read_text()) for f in a.measurements), key=lambda r: r["seed"])
    assert len({r["seed"] for r in results}) == 3 and all(r["device"] == results[0]["device"] == "NVIDIA L4" for r in results)
    assert all(r["source_sha256"] == results[0]["source_sha256"] for r in results)
    parsed = [analyze(record) for record in results]
    headline = {}
    for geometry in GEOMETRIES:
        modes = [r["arms"][geometry] for r in parsed]
        headline[geometry] = {
            "fresh_proc_1024": {timing: {arm: mean(m["procedural"][timing]["1024"][arm] /
                                                         m["procedural"][timing]["1024"]["questions"] for m in modes)
                                        for arm in CONDITIONS} for timing in CONDITIONS},
            "archive_48": {tag: {arm: mean(m["archive"][tag][48]["joint"][arm] /
                                                m["archive"][tag][48]["joint"]["questions"] for m in modes)
                                 for arm in CONDITIONS} for tag in ("earlier", "later")},
        }
    data = {"source_sha256": results[0]["source_sha256"], "headline": headline, "by_seed": parsed}
    a.output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
