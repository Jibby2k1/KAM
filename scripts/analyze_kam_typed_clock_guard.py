#!/usr/bin/env python3
"""Audit hard chronological visibility and local rank on identical held-out histories."""

import argparse
import json
from pathlib import Path
from statistics import median

from analyze_kam_clock_archive_controls import estimate
from analyze_kam_native_musique import compare
from analyze_kam_temporal_projection_control import clean_cases, read

CONTRASTS = {"timestamps": (), "timestamps_and_rank": ("timestamps",), "hard_clock": ("timestamps",),
             "hard_clock_and_rank": ("timestamps", "timestamps_and_rank", "hard_clock")}
POLICIES = tuple(CONTRASTS)


def measure(runs, geometry):
    output = {}
    for arm in POLICIES:
        label = f"{geometry}/{arm}"
        timing = median(r["arms"][label]["fit"]["seconds"] for r in runs)
        natural, synthetic = {}, {}
        for topic in runs[0]["sources"]:
            natural[topic] = {}
            for depth in (16, 48):
                panels = {p: clean_cases(runs, f"{geometry}/{p}", topic, depth) for p in POLICIES}
                actual = panels[arm]
                assert all(p.keys() == actual.keys() for p in panels.values())
                natural[topic][depth] = {
                    "n": len(actual), "strict": sum(v["joint"] for v in actual.values()),
                    "numeric": sum(v["numeric_exact"] for v in actual.values()),
                    "paired_strict": {p: estimate(panels[p], actual, "joint", months=topic != "dallas")
                                      for p in CONTRASTS[arm]},
                }
        for variant in ("original", "corrections"):
            synthetic[variant] = {}
            for length in ("16", "48", "128", "512", "1024"):
                checks = {}
                for r in runs:
                    stored = {p: r["arms"][f"{geometry}/{p}"]["generated"][variant][length] for p in POLICIES}
                    assert len({entry["fixture_sha256"] for entry in stored.values()}) == 1
                    actual = [all(x) for x in stored[arm]["outcomes"]]
                    groups = [[i] for i in range(len(actual))]
                    checks[r["seed"]] = {"n": len(actual), "strict": sum(actual), "paired": {
                        p: compare(actual, [all(x) for x in stored[p]["outcomes"]], groups) for p in CONTRASTS[arm]}}
                synthetic[variant][length] = checks
        output[arm] = {"fit_median_s": timing, "natural": natural, "procedural": synthetic}
    return output


def run(seeds):
    assert len(seeds) == len({r["seed"] for r in seeds}) == 3
    assert {r["device"] for r in seeds} == {"NVIDIA L4"}
    assert all((r["clock"], r["steps"], r["token_count"]) == ("log_available_mean", 3200, 10556800) for r in seeds)
    assert all(r["sources"] == seeds[0]["sources"] for r in seeds)
    counts = {}
    for r in seeds:
        assert set(r["arms"]) == {f"{g}/{p}" for g in ("dot", "radial") for p in POLICIES}
    for g in ("dot", "radial"):
        for policy in POLICIES:
            if policy.startswith("hard_clock"):
                counts[f"{g}/{policy}"] = [r["arms"][f"{g}/{policy}"]["audit"] for r in seeds]
                assert all(a["exact_hidden_suffix_null_effect"] == 32 and a["excluded"] > 0 for a in counts[f"{g}/{policy}"])
    return {"sources": seeds[0]["sources"], "null_effect_audit": counts,
            "outcomes": {g: measure(seeds, g) for g in ("dot", "radial")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = sorted(map(read, a.measurements), key=lambda r: r["seed"])
    a.output.write_text(json.dumps(run(rows), sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
