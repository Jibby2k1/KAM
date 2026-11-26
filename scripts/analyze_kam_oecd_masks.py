#!/usr/bin/env python3
"""Analyze paired temporal masking interventions on the Dallas Fed archive."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

from analyze_kam_oecd_revisions import ARMS, paired_quarter_seed

PANELS = ("observed", "three_releases", "load_control")
FIELDS = ("arm", "seed", "train_tokens", "train_input_sha256", "archive_fixture_sha256", "device", "archive_masks")


def extract(run):
    original = {(r["id"], r["period"], r["recode"]): r["joint"]
                for r in run["archive"] if r["depth"] == 48}
    observed = {(group, period, recode): hit
                for group, period, _, recode, clock, panel, _, hit in run["archive_masks"]
                if clock == 1 and panel == "observed"}
    assert original == observed
    return {k: run[k] for k in FIELDS} | {"unmasked_parity_sha256": sha256(bytes(original.values())).hexdigest()}


def summarize(runs, provenance):
    assert len(runs) == 8
    seeds = sorted({r["seed"] for r in runs})
    assert len(seeds) == 4
    assert len({r["device"] for r in runs}) == 1
    assert {r["train_tokens"] for r in runs} == {13516800}
    assert {r["archive_fixture_sha256"] for r in runs} == {provenance["archive_fixture_sha256"]}
    data, counts, fits = {}, {}, {}
    for run in runs:
        arm, seed = run["arm"], run["seed"]
        assert arm in ARMS
        assert (arm, seed) not in fits
        fits[arm, seed] = run["train_input_sha256"]
        for group, period, depth, recode, clock, panel, count, hit in run["archive_masks"]:
            assert depth == 48 and clock in (1, 3) and panel in PANELS and recode in (0, 1)
            index = (arm, clock, panel)
            key = (seed, group, period, recode)
            indexed = data.setdefault(index, {})
            assert key not in indexed
            indexed[key] = int(hit)
            counts[index, key] = count
    for seed in seeds:
        assert fits[ARMS[0], seed] == fits[ARMS[1], seed]
    keys = data[ARMS[0], 1, "observed"].keys()
    assert len(keys) == 16 * sum(provenance["counts"].values())
    assert all(panel.keys() == keys for panel in data.values())
    assert all(counts[((a, 1, "observed"), k)] == 48 for a in ARMS for k in keys)
    for key in keys:
        visible = {counts[((a, t, p), key)] for a in ARMS for t in (1, 3) for p in PANELS[1:]}
        assert len(visible) == 1 and 35 <= next(iter(visible)) < 48
    active = [counts[((ARMS[0], 1, "three_releases"), k)] for k in keys]

    def delta(a, b):
        return {k: a[k] - b[k] for k in keys}

    performance = {f"{a}:{t}:{p}": {"joint": sum(data[a, t, p].values()), "n": len(keys)}
                   for a in ARMS for t in (1, 3) for p in PANELS}
    effects = {}
    cluster_diagnostics = {}
    for arm in ARMS:
        for clock in (1, 3):
            observed, same, load = (data[arm, clock, p] for p in PANELS)
            for label, a, b in (("three_vs_observed", same, observed), ("load_vs_observed", load, observed),
                                ("three_vs_load", same, load)):
                effects[f"{arm}:{clock}:{label}"] = paired_quarter_seed(delta(a, b))
            for low, high in ((4, 6), (7, 9), (10, 11)):
                group = [k for k in keys if low <= 51 - counts[((arm, clock, "three_releases"), k)] <= high]
                if group:
                    cluster_diagnostics[f"{arm}:{clock}:{low}-{high}"] = {
                        "n": len(group), "observations": len({k[1] for k in group}),
                        "three_minus_load": mean(same[k] - load[k] for k in group),
                    }
        effects[f"{arm}:clock_by_mask"] = paired_quarter_seed(delta(
            delta(data[arm, 3, "three_releases"], data[arm, 3, "load_control"]),
            delta(data[arm, 1, "three_releases"], data[arm, 1, "load_control"]),
        ))
    return {"device": runs[0]["device"], "fixture_sha256": runs[0]["archive_fixture_sha256"],
            "seeds": seeds, "groups": provenance["counts"], "candidate_slots": 48,
            "masked_active_facts": {"min": min(active), "max": max(active), "median": median(active),
                                    "mean": mean(active)},
            "performance": performance, "effects": effects, "exploratory_cluster_bins": cluster_diagnostics}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--provenance", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--write-replay", type=Path)
    a = p.parse_args()
    if a.input.is_dir():
        runs = [extract(json.loads(p.read_text())) for p in sorted(a.input.glob("eval-*.json"))]
    else:
        runs = json.loads(gzip.decompress(a.input.read_bytes()))
    provenance = json.loads(a.provenance.read_text())
    output = summarize(runs, provenance)
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        a.write_replay.write_bytes(gzip.compress(json.dumps(runs, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
