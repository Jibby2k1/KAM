#!/usr/bin/env python3
"""Audit paired independent archive runs and summarize transfer and interventions."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from random import Random
from statistics import mean

ARMS = ("dot_without_absolute", "radial_without_absolute")
SOURCES = ("procedural_dual_boundary", "procedural_archive_topology", "archive")
PANELS = ("archive", "archive_scaled", "archive_thin", "archive_thin_scaled")
RUN_FIELDS = ("seed", "arm", "training_data", "train_tokens", "device", "archive_fixture_sha256", "archive_thin_sha256")
ROW_FIELDS = ("id", "series", "depth", "period", "recode", "joint", "numeric_exact")


def paired_quarter_seed(differences):
    blocks = defaultdict(list)
    for (seed, group, _, _), difference in differences.items():
        blocks[seed, int(group.rsplit("-", 1)[1])].append(difference)
    quarters = sorted({q for _, q in blocks})
    seeds = sorted({s for s, _ in blocks})
    assert len(blocks) == len(quarters) * len(seeds)
    totals = {key: (sum(values), len(values)) for key, values in blocks.items()}
    rng = Random(498612)
    replicates = []
    for _ in range(5000):
        fit_seeds = rng.choices(seeds, k=len(seeds))
        sampled_quarters = rng.choices(quarters, k=len(quarters))
        selected = [totals[s, q] for s in fit_seeds for q in sampled_quarters]
        replicates.append(sum(d for d, _ in selected) / sum(n for _, n in selected))
    replicates.sort()
    return {"delta": mean(differences.values()), "bootstrap95": [replicates[125], replicates[4874]],
            "quarter_clusters": len(quarters), "fit_seeds": len(seeds)}


def summarize(raw, provenance):
    assert len(raw) == 24 and len({r["seed"] for r in raw}) == 4
    assert {r["train_tokens"] for r in raw} == {13516800}
    assert len({r["device"] for r in raw}) == 1
    for field in ("archive_fixture_sha256", "archive_thin_sha256"):
        assert all(r[field] == provenance[field] for r in raw)
    counts = provenance["counts"]
    n = sum(counts.values())
    tests = {}
    for depth in (16, 48):
        indexed = {}
        for a in ARMS:
            for source in SOURCES:
                selected = [r for r in raw if r["training_data"] == source and r["arm"] == a]
                assert len(selected) == 4
                for panel in PANELS:
                    rows = [(run["seed"], r) for run in selected for r in run[panel] if r["depth"] == depth]
                    items = {(seed, r["id"], r["period"], r["recode"]): r for seed, r in rows}
                    assert len(rows) == len(items) == 16*n
                    assert Counter(r["series"] for _, r in rows) == {k: v*16 for k,v in counts.items()}
                    indexed[a, source, panel] = items
        keys = next(iter(indexed.values())).keys()
        assert all(x.keys() == keys for x in indexed.values())
        performance = {}
        effects = {}
        for (arm, source, panel), items in indexed.items():
            name = f"{arm}:{source}:{panel}"
            performance[name] = {country: {"n": len(group), "joint": sum(r["joint"] for r in group),
                                           "numerical": sum(r["numeric_exact"] for r in group)}
                                 for country in ("all", *counts)
                                 if (group := [r for r in items.values() if country == "all" or r["series"] == country])}
            if panel == "archive":
                for intervention in PANELS[1:]:
                    other = indexed[arm, source, intervention]
                    effects[f"{arm}:{source}:{intervention}_minus_observed"] = paired_quarter_seed(
                        {k: int(other[k]["joint"]) - int(items[k]["joint"]) for k in keys})
        for arm in ARMS:
            for panel in PANELS:
                topo = indexed[arm, "procedural_archive_topology", panel]
                boundary = indexed[arm, "procedural_dual_boundary", panel]
                effects[f"{arm}:{panel}:release_clusters_minus_boundary"] = paired_quarter_seed(
                    {k: int(topo[k]["joint"]) - int(boundary[k]["joint"]) for k in keys})
        tests[depth] = {"performance": performance, "effects": effects}
    return {"sources": provenance["sources"], "groups": counts, "seeds": sorted({r["seed"] for r in raw}),
            "train_tokens": raw[0]["train_tokens"], "oracle": provenance["oracle"], "tests": tests,
            **{k: raw[0][k] for k in ("archive_fixture_sha256", "archive_thin_sha256", "device")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="A directory of HPG results or a compressed replay bundle")
    p.add_argument("--provenance", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--write-replay", type=Path)
    a = p.parse_args()
    if a.input.is_dir():
        raw = [json.loads(p.read_text()) for p in sorted(a.input.glob("eval-*.json"))]
    else:
        raw = json.loads(gzip.decompress(a.input.read_bytes()))
    provenance = json.loads(a.provenance.read_text())
    summary = summarize(raw, provenance)
    a.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        small = [{k: run[k] for k in RUN_FIELDS} |
                 {panel: [{k: row[k] for k in ROW_FIELDS} for row in run[panel]] for panel in PANELS}
                 for run in raw]
        a.write_replay.write_bytes(gzip.compress(json.dumps(small, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
