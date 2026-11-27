#!/usr/bin/env python3
"""Audit from-scratch radial precision training and archived revision transfer."""

import argparse
import json
from pathlib import Path
from statistics import mean

from analyze_kam_shared_serving import unpack

MODES = ("eager_fp32", "eager_bf16", "augmented_flash_bf16", "balanced_split_flash_bf16")
SEEDS = (931121, 931157, 931187)
STEPS = 3200
TOPICS = {"earlier": 160, "later": 32}
DEPTHS = (16, 48)
OUTPUTS = ("joint", "numeric_exact", "unbound")
EVALS = ("common_fp32_eval", "native_eval")


def fingerprint_panels(panels, cohort):
    result, identities = {}, {}
    assert [(p["depth"], p["recode"]) for p in panels] == [(d, r) for d in DEPTHS for r in (0, 1)]
    ids = set()
    for p in panels:
        key = p["depth"], p["recode"]
        rows = p["outcomes"]
        identities[key] = [(x["id"], x["period"], x["series"]) for x in rows]
        assert len(rows) == len(set(identities[key])) == TOPICS[cohort] * 2
        ids.update(x["id"] for x in rows)
        assert all(all(isinstance(x[k], bool) for k in OUTPUTS) for x in rows)
        result[key] = p["fixture_sha256"]
    assert len(ids) == TOPICS[cohort]
    assert all(x == identities[(16, 0)] for x in identities.values())
    return result, identities[(16, 0)]


def inspect(run):
    assert run["schedule"] == [[16, 159], [48, 33]] and run["bank_count"] == 16_384
    assert tuple(run["modes"]) == (*MODES[run["order"]:], *MODES[:run["order"]])
    base = run["modes"]["eager_fp32"]
    assert base["common_fp32_eval"] == base["native_eval"]
    expected = {k: fingerprint_panels(base["common_fp32_eval"][k], k) for k in TOPICS}
    group_ids = {k: {i for i, _, _ in val[1]} for k, val in expected.items()}
    assert group_ids["earlier"].isdisjoint(group_ids["later"])
    metrics = {}
    for mode in MODES:
        curr = run["modes"][mode]
        fit = curr["training"]
        assert fit["steps"] == STEPS and fit["tokens"] == 13_516_800 and fit["elapsed_seconds"] > 0
        assert fit["peak_extra_allocated_bytes"] > 0
        track = fit["sampled_losses"]
        assert [point["step"] for point in track] == [100, 400, 800, 1600, STEPS]
        assert all(0 <= point["loss"] < float("inf") for point in track)
        assert [point["tokens"] for point in track] == [point["step"] * 4224 for point in track]
        metrics[mode] = {"training": fit, "training_tokens_per_second": round(fit["tokens"] / fit["elapsed_seconds"], 4)}
        for precision in EVALS:
            outcomes = curr[precision]
            assert set(outcomes) == set(TOPICS)
            cohort_stats = {}
            for cohort, panels in outcomes.items():
                assert fingerprint_panels(panels, cohort) == expected[cohort]
                cohort_stats[cohort] = {}
                for depth in DEPTHS:
                    rows = [r for panel in panels if panel["depth"] == depth for r in panel["outcomes"]]
                    cohort_stats[cohort][str(depth)] = {"n": len(rows), **{k: sum(r[k] for r in rows) for k in OUTPUTS}}
            metrics[mode][precision] = cohort_stats
    stamp = tuple(run[k] for k in ("start_sha256", "bank_sha256", "index_sha256", "source_sha256", "order")), expected
    return metrics, stamp


def summarize(runs):
    by_gpu, anchors = {}, {}
    for hardware in ("b200", "l4"):
        records = {r["seed"]: r for r in runs[hardware]}
        assert set(records) == set(SEEDS) and len(runs[hardware]) == len(SEEDS)
        assert all(r["order"] == i for i, r in enumerate(records[s] for s in SEEDS))
        assert all(r["device"] == ("NVIDIA B200" if hardware == "b200" else "NVIDIA L4") for r in records.values())
        seeds = {}
        for s in SEEDS:
            seeds[str(s)], stamp = inspect(records[s])
            assert anchors.setdefault(s, stamp) == stamp
        totals = {}
        for mode in MODES:
            totals[mode] = {
                precision: {
                    cohort: {str(d): {
                        method: round(mean(seed[mode][precision][cohort][str(d)][method] /
                                           seed[mode][precision][cohort][str(d)]["n"] for seed in seeds.values()), 8)
                        for method in OUTPUTS}
                        for d in DEPTHS}
                    for cohort in TOPICS}
                for precision in EVALS}
            totals[mode]["mean_training_seconds"] = round(mean(seed[mode]["training"]["elapsed_seconds"]
                                                               for seed in seeds.values()), 6)
        by_gpu[hardware] = {"seeds": seeds, "means": totals}
    return by_gpu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b200", required=True, type=Path, nargs=3)
    p.add_argument("--l4", required=True, type=Path, nargs=3)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    summary = summarize({k: [unpack(f) for f in getattr(a, k)] for k in ("b200", "l4")})
    a.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
