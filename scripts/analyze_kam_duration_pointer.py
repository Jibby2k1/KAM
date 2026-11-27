#!/usr/bin/env python3
"""Audit learned event pointers on matched synthetic and published schedules."""

import argparse
import json
from pathlib import Path
from statistics import mean

from analyze_kam_shared_serving import unpack
from analyze_kam_tempreplay_swaps import SCENARIOS

MODES = ("local", "dot", "radial", "mean_duration", "exact_prefix")
SEEDS = (452671, 452701, 452731)
CONDITIONS = ("16/90", "32/90", "64/90", "128/90", "64/45", "128/22")


def compare_archive(cases, records):
    counts = {m: dict.fromkeys((*SCENARIOS, "correct_before_and_after_intervention", "correct_before_and_after_sham",
                               "changed_on_intervention", "changed_on_sham"), 0) for m in MODES}
    for expected, result in zip(cases, records, strict=True):
        outcomes = result["scenarios"]
        assert set(outcomes) == set(SCENARIOS) and all(set(x) == set(MODES) for x in outcomes.values())
        gold = {s: expected["variants"][s]["gold"] for s in SCENARIOS}
        assert len({outcomes[s]["mean_duration"] for s in SCENARIOS}) == 1
        for m, stats in counts.items():
            chosen = {s: outcomes[s][m] for s in SCENARIOS}
            assert all(expected["anchor"] <= pred < expected["events"] for pred in chosen.values())
            for s in SCENARIOS:
                stats[s] += chosen[s] == gold[s]
            for s in ("sham", "intervention"):
                stats[f"correct_before_and_after_{s}"] += chosen["original"] == gold["original"] and chosen[s] == gold[s]
                stats[f"changed_on_{s}"] += chosen["original"] != chosen[s]
    assert counts["exact_prefix"]["correct_before_and_after_intervention"] == len(cases)
    assert counts["mean_duration"]["correct_before_and_after_intervention"] == 0
    return counts


def inspect(raw, fixture):
    assert raw["device"] == "NVIDIA L4"
    assert raw["fixture_sha256"] == fixture["fixture_sha256"]
    assert raw["steps"] == 2400 and raw["bank"] == 16_384
    assert set(raw["training"]) == set(MODES[:3])
    for fit in raw["training"].values():
        assert fit["tokens"] == 2_572_800 and fit["seconds"] > 0
        assert 60_000 < fit["parameters"] < 90_000
        assert [item["step"] for item in fit["losses"]] == [100, 400, 800, 1600, 2400]
        assert all(0 <= item["loss"] < float("inf") for item in fit["losses"])

    assert set(raw["synthetic"]) == set(CONDITIONS)
    synthetic = {}
    for shape, panel in raw["synthetic"].items():
        events = int(shape.split("/")[0])
        assert len(panel["gold"]) == 512 and set(panel["predictions"]) == set(MODES)
        assert panel["predictions"]["exact_prefix"] == panel["gold"]
        assert all(0 <= predicted < events for z in panel["predictions"].values() for predicted in z)
        synthetic[shape] = {k: sum(a == b for a, b in zip(chosen, panel["gold"], strict=True))
                            for k, chosen in panel["predictions"].items()}

    cases, archived = fixture["rows"], raw["archived"]
    assert len(cases) == 1173 and [r["id"] for r in cases] == [r["id"] for r in archived]
    totals = compare_archive(cases, archived)
    plans = sorted({r["plan"] for r in cases})
    by_plan = {}
    for plan in plans:
        chosen = [(c, r) for c, r in zip(cases, archived, strict=True) if c["plan"] == plan]
        cohort, predictions = zip(*chosen, strict=True)
        by_plan[plan] = {"n": len(cohort), "correct": compare_archive(cohort, predictions)}
    return {"training": raw["training"], "synthetic_correct_of_512": synthetic,
            "archived_correct_of_1173": totals, "archived_by_plan": by_plan}


def summarize(runs, fixture):
    records = {r["seed"]: r for r in runs}
    assert len(runs) == len(records) == len(SEEDS) and set(records) == set(SEEDS)
    assert len({tuple(r["bank_sha256"]) for r in runs}) == len(SEEDS)
    for shape in CONDITIONS:
        assert len({r["synthetic"][shape]["sha256"] for r in runs}) == len(SEEDS)
    details = {str(seed): inspect(records[seed], fixture) for seed in SEEDS}
    archive_mean = {m: {metric: round(mean(x["archived_correct_of_1173"][m][metric] / 1173 for x in details.values()), 8)
                        for metric in details[str(SEEDS[0])]["archived_correct_of_1173"][m]} for m in MODES}
    synthetic_mean = {s: {m: round(mean(x["synthetic_correct_of_512"][s][m] / 512 for x in details.values()), 8)
                          for m in MODES} for s in CONDITIONS}
    return {"per_seed": details, "mean_archived_rates": archive_mean, "mean_synthetic_accuracy": synthetic_mean}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs=3, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    payload = summarize([unpack(f) for f in args.runs], unpack(args.manifest))
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
