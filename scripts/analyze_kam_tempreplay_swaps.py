#!/usr/bin/env python3
"""Summarize evidence recovery on matched temporal duration interventions."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from analyze_kam_shared_serving import unpack

SCENARIOS = ("original", "sham", "intervention")
MODES = ("direct", "target_clause", "window", "window_target_clause", "duration_masked")


def selectors(cases, predictions):
    count = {}
    for mode in MODES:
        stats = Counter()
        for expected, raw in zip(cases, predictions, strict=True):
            outcomes = raw["scenarios"]
            gold = {s: expected["variants"][s]["gold"] for s in SCENARIOS}
            chosen = {s: outcomes[s][mode]["prediction"] for s in SCENARIOS}
            ranks = {s: outcomes[s][mode]["gold_rank"] for s in SCENARIOS}
            for s in SCENARIOS:
                assert 0 <= chosen[s] < expected["events"] and 1 <= ranks[s] <= expected["events"]
                assert (ranks[s] == 1) == (chosen[s] == gold[s])
                stats[f"{s}_top1"] += ranks[s] == 1
                stats[f"{s}_top5"] += ranks[s] <= 5
            if mode == "duration_masked":
                assert len(set(chosen.values())) == 1
            stats["both_original_and_intervention_top1"] += ranks["original"] == ranks["intervention"] == 1
            stats["both_original_and_sham_top1"] += ranks["original"] == ranks["sham"] == 1
            stats["changed_prediction_sham"] += chosen["original"] != chosen["sham"]
            stats["changed_prediction_intervention"] += chosen["original"] != chosen["intervention"]
        count[mode] = dict(stats)
    return count


def summarize(fixture, results):
    rows = fixture["rows"]
    assert len(rows) == len({r["id"] for r in rows}) == 1173
    assert sorted(s["shard"] for s in results) == list(range(4))
    assert all(s["shards"] == 4 and s["fixture_sha256"] == fixture["fixture_sha256"] for s in results)
    assert all(s["models"] == results[0]["models"] and s["device"] == "NVIDIA L4" for s in results)
    for s in results:
        assert [r["id"] for r in s["rows"]] == [r["id"] for r in rows[s["shard"]::4]]
    outcomes = {r["id"]: r for shard in results for r in shard["rows"]}
    for row in rows:
        v = row["variants"]
        assert v["original"]["gold"] == v["sham"]["gold"] != v["intervention"]["gold"]
        assert v["original"]["labels"] == v["sham"]["labels"] != v["intervention"]["labels"]
        control, test = v["sham"]["swap"], v["intervention"]["swap"]
        assert max(v["original"]["gold"], v["intervention"]["gold"]) < min(control)
        assert row["anchor"] <= test[0] < v["original"]["gold"] < test[1] < row["events"]
        assert len(control) == len(set(control)) == len(test) == len(set(test)) == 2
        assert len(set(control) | set(test)) >= 3
        actual = outcomes[row["id"]]["scenarios"]
        assert set(actual) == set(SCENARIOS) and all(set(x) == set(MODES) for x in actual.values())
    subsets = {"all": rows}
    subsets.update({f"type{q}": [row for row in rows if row["kind"] == q] for q in (1, 2)})
    subsets.update({f"plan/{g}": [row for row in rows if row["plan"] == g] for g in sorted({r["plan"] for r in rows})})
    return {"fixture_sha256": fixture["fixture_sha256"], "source_counts": fixture["counts"], "plans": 10,
            "swap_magnitude_minutes": {"minimum": min(r["absolute_delta_minutes"] for r in rows),
                                       "median": median(r["absolute_delta_minutes"] for r in rows),
                                       "maximum": max(r["absolute_delta_minutes"] for r in rows)},
            "slices": {k: {"questions": len(cases), "policies": selectors(cases, [outcomes[r["id"]] for r in cases])}
                       for k, cases in subsets.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evaluations", nargs=4, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    summary = summarize(unpack(args.manifest), [unpack(f) for f in args.evaluations])
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
