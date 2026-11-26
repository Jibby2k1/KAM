#!/usr/bin/env python3
"""Audit paired-period and paired-noise TimeQA model results."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_kam_fullwiki_support_links import paired
from audit_kam_natural_data import digest

ARMS = ("dense_no_date", "dense", "learned_no_dates", "learned_dates")
SCENARIOS = ("clean", "matched_noise")


def counts(rows, scenario, arm):
    groups = [[q["scenarios"][scenario][arm] for q in row["questions"]] for row in rows]
    flat = [r for group in groups for r in group]
    return {
        "evidence_hit": sum(r["evidence_hit"] for r in flat),
        "answer_em": sum(r["em"] for r in flat),
        "joint": sum(r["em"] and r["evidence_hit"] for r in flat),
        "opposite_period": sum(r["opposite_period"] for r in flat),
        "opposite_answer": sum(r["opposite_answer_em"] for r in flat),
        "distractor": sum(r["distractor"] for r in flat),
        "both_periods_evidence": sum(all(r["evidence_hit"] for r in g) for g in groups),
        "both_periods_answer": sum(all(r["em"] for r in g) for g in groups),
    }


def compare(rows, scenario_a, arm_a, scenario_b, arm_b):
    differences = {metric: [] for metric in ("evidence_hit", "em", "joint", "both_periods_evidence")}
    for row in rows:
        a = [q["scenarios"][scenario_a][arm_a] for q in row["questions"]]
        b = [q["scenarios"][scenario_b][arm_b] for q in row["questions"]]
        for metric in ("evidence_hit", "em"):
            differences[metric].append((sum(r[metric] for r in a) - sum(r[metric] for r in b)) / 2)
        differences["joint"].append((sum(r["em"] and r["evidence_hit"] for r in a) -
                                     sum(r["em"] and r["evidence_hit"] for r in b)) / 2)
        differences["both_periods_evidence"].append(all(r["evidence_hit"] for r in a) -
                                                    all(r["evidence_hit"] for r in b))
    return {metric: paired(diff) for metric, diff in differences.items()}


def summarize(data, fit, shards, sha):
    assert all(r["fixture_sha256"] == sha for r in [fit, *shards])
    assert all(r["weights_sha256"] == fit["weights_sha256"] and r["models"] == fit["models"] for r in shards)
    assert [r["id"] for r in fit["rows"]] == [r["id"] for r in data["rows"]["dev"]]
    subset = data["rows"]["test"]
    rows = []
    for i, shard in enumerate(shards):
        expected = subset[i::len(shards)]
        assert [r["id"] for r in shard["rows"]] == [r["id"] for r in expected]
        for record, fixture in zip(shard["rows"], expected, strict=True):
            context = fixture["context"]
            assert record["clean_depth"] == len(context)
            assert record["kind"] == fixture["type"]
            assert fixture["questions"][0]["upper"] < fixture["questions"][1]["lower"]
            assert not set(fixture["questions"][0]["gold"]) & set(fixture["questions"][1]["gold"])
            for j, question in enumerate(record["questions"]):
                gold = set(fixture["questions"][j]["gold"])
                other = set(fixture["questions"][1 - j]["gold"])
                assert question["gold"] == sorted(gold)
                for scenario, arm_outputs in question["scenarios"].items():
                    assert set(arm_outputs) == set(ARMS)
                    cap = len(context) + (len(fixture["noise"]) if scenario == "matched_noise" else 0)
                    for output in arm_outputs.values():
                        pick = output["selected"]
                        assert 0 <= pick < cap
                        assert output["evidence_hit"] == (pick in gold)
                        assert output["opposite_period"] == (pick in other)
                        assert output["distractor"] == (pick >= len(context))
            rows.append(record)

    comparisons = {}
    for scenario in SCENARIOS:
        comparisons[scenario] = {
            "dated_dense_vs_undated": compare(rows, scenario, "dense", scenario, "dense_no_date"),
            "learned_date_vs_no_date": compare(rows, scenario, "learned_dates", scenario, "learned_no_dates"),
            "learned_date_vs_dense": compare(rows, scenario, "learned_dates", scenario, "dense"),
        }
    comparisons["effect_of_distractors"] = {arm: compare(rows, "matched_noise", arm, "clean", arm)
                                             for arm in ARMS}
    return {"recruitment": data["counts"], "entities": len(rows), "questions": len(rows) * 2,
            "property_counts": dict(Counter(r["kind"] for r in rows)),
            "oracle_answer_em": sum(q["oracle"]["em"] for r in rows for q in r["questions"]),
            "scenarios": {s: {a: counts(rows, s, a) for a in ARMS} for s in SCENARIOS},
            "comparisons": comparisons,
            "fit_seconds": fit["elapsed_seconds"], "eval_seconds": [s["elapsed_seconds"] for s in shards],
            "fixture_sha256": sha, "weights_sha256": fit["weights_sha256"], "models": fit["models"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("shards", nargs="+", type=Path)
    a = p.parse_args()
    data = summarize(json.loads(a.fixture.read_text()), json.loads(a.fit.read_text()),
                     [json.loads(p.read_text()) for p in a.shards], digest(a.fixture))
    a.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
