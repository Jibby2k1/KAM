#!/usr/bin/env python3
"""Verify exact clock-cache trajectories and report paired update time by GPU."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median

GEOMETRIES = ("dot", "radial")
EAGER = ("on_the_fly", "cached_uint8")
EXPECTED_SEEDS = {791513, 791543, 791553}


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def time_pair(before, after, setup=None):
    result = {"reference_s": median(before), "alternative_s": median(after),
              "paired_changes_s": [b-a for a, b in zip(before, after, strict=True)]}
    if setup:
        result["alternative_total_s"] = median(b + cost for b, cost in zip(after, setup, strict=True))
    return result


def summary(runs, graph=False):
    batches = {}
    for row in runs:
        batches.setdefault(row["device"], []).append(row)
    results = {}
    for device, trials in batches.items():
        assert {r["seed"] for r in trials} == EXPECTED_SEEDS and len(trials) == len(EXPECTED_SEEDS)
        trials.sort(key=lambda r: r["seed"])
        assert all(r["cache_bytes"] == 1064960 for r in trials)
        fixed = [r["cache_build_seconds" if graph else "setup_seconds"] for r in trials]
        by_model = {}
        for model in GEOMETRIES:
            if graph:
                controls = {k: [] for k in (
                    "eager/on_the_fly", "eager/cached_uint8", "shape_graph/on_the_fly", "shape_graph/cached_uint8")}
                for row in trials:
                    values = row["arms"][model]
                    assert set(values["runs"]) == set(values["parity"]) == set(controls)
                    for name, audit in values["parity"].items():
                        assert audit["weights"]["equal"] and audit["adamw_moments"]["equal"]
                        assert audit["same_loss"] and audit["same_decisions"]
                        controls[name].append(values["runs"][name])
                timing = {k: {"median_update_s": median(x["wall_seconds"] for x in v),
                              "median_including_graph_setup_s": median(x["wall_seconds"] + x["setup_seconds"] for x in v),
                              "median_including_all_setup_s": median(x["wall_seconds"] + x["setup_seconds"] + cost
                                                                     for x, cost in zip(v, fixed if "cached" in k else [0.]*3,
                                                                                        strict=True)),
                              "median_peak_bytes": median(x["peak_extra_allocated_bytes"] for x in v)}
                          for k, v in controls.items()}
                comparisons = {k: time_pair([x["wall_seconds"] for x in controls["eager/on_the_fly"]],
                                            [x["wall_seconds"] for x in v]) for k, v in controls.items()}
                by_model[model] = {"timing": timing, "against_eager_online": comparisons}
            else:
                a, b = [ [r["results"][f"{model}/{m}"] for r in trials] for m in EAGER]
                for x, y in zip(a, b, strict=True):
                    assert x["heldout"] == y["heldout"] and x["state_sha256"] == y["state_sha256"]
                    assert x["training"]["optimizer_sha256"] == y["training"]["optimizer_sha256"]
                    assert x["training"]["losses"] == y["training"]["losses"]
                by_model[model] = time_pair([x["training"]["seconds"] for x in a],
                                            [y["training"]["seconds"] for y in b], fixed)
        results[device] = {"measurements": len(trials), "cache_size_bytes": trials[0]["cache_bytes"],
                           "cache_build_seconds": fixed, "geometries": by_model}
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eager", nargs="+", type=Path, required=True)
    p.add_argument("--graph", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = {"eager": summary(list(map(read, a.eager)))}
    if a.graph:
        result["graph"] = summary(list(map(read, a.graph)), graph=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
