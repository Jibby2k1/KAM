#!/usr/bin/env python3
"""Verify TimelineQA monthly count support and matched date-collision effects."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from analyze_kam_native_musique import compare
from analyze_kam_shared_serving import unpack

DEPTHS = (4, 8, 16, 32, 64)
MODES = ("dense", "shown_month", "source_month")
LITERALS = ("full_literal_shown", "full_literal_source")


def reference(case, experiment):
    q = case
    relevant = set(q["gold"])
    candidates = set(q["candidates"])
    donors = set(q["changed_dates"])
    observations = {}
    for arm, trial in experiment["arms"].items():
        typed = set(trial["source_month"])
        shown = set(trial["shown_month"])
        assert relevant <= typed <= candidates
        assert donors.isdisjoint(typed)
        assert shown == (typed if arm == "sham" else typed | donors)
        pool = {"dense": candidates, "shown_month": shown, "source_month": typed}
        read = {}
        for mode, ids in trial["ranks"].items():
            assert len(ids) == len(set(ids)) and set(ids) == pool[mode]
            for depth in DEPTHS:
                top = set(ids[:depth])
                count = len(top & relevant)
                read[f"{mode}/{depth}"] = {"found": count, "all_support": relevant <= top,
                                            "donors": len(top & donors), "truth": len(relevant)}
        for mode in LITERALS:
            selected = set(trial[mode])
            assert selected <= (shown if mode.endswith("shown") else typed)
            read[mode] = {"count": len(selected), "set_exact": selected == relevant,
                          "count_exact": len(selected) == len(relevant), "false_donors": len(selected & donors),
                          "error": len(selected) - len(relevant)}
        observations[arm] = read
    assert experiment["arms"]["sham"]["ranks"]["source_month"] == experiment["arms"]["collision"]["ranks"]["source_month"]
    assert observations["sham"]["full_literal_source"] == observations["collision"]["full_literal_source"]
    return observations


def summarize(cases):
    grouped = defaultdict(list)
    for i, (persona, _, _) in enumerate(cases):
        grouped[persona].append(i)
    flags = {"recovery": "all_support", "literal_count": "count_exact"}
    measurements = {}
    for mode in cases[0][-1]["sham"]:
        what = "literal_count" if mode in LITERALS else "recovery"
        column = flags[what]
        sham, altered = [[m[arm][mode] for _, _, m in cases] for arm in ("sham", "collision")]
        measurements[mode] = {
            "sham": {k: mean(r[k] for r in sham) for k in sham[0]},
            "collision": {k: mean(r[k] for r in altered) for k in altered[0]},
            "collision_change": compare([r[column] for r in altered], [r[column] for r in sham], list(grouped.values())),
            "success_key": column,
        }
    return {"questions": len(cases), "personas": len(grouped), "results": measurements}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--measurements", type=Path, nargs=4, required=True)
    a = p.parse_args()
    data = unpack(a.manifest)
    results = sorted(map(unpack, a.measurements), key=lambda p: p["shard"])
    assert [r["shard"] for r in results] == list(range(4))
    seen = []
    for r in results:
        assert r["fixture_sha256"] == data["fixture_sha256"] and r["gpu"] == "NVIDIA L4" and r["shards"] == 4
        assert r["models"] == results[0]["models"]
        for ranking, fixture in zip(r["personas"], data["personas"][r["shard"]::4], strict=True):
            assert ranking["seed"] == fixture["seed"]
            for trial, case in zip(ranking["rows"], fixture["cases"], strict=True):
                assert trial["id"] == case["id"]
                seen.append((fixture["seed"], case, reference(case, trial)))
    assert len(seen) == sum(len(r["cases"]) for r in data["personas"])
    scopes = {"all": seen, "positive": [t for t in seen if t[1]["gold"]], "zero": [t for t in seen if not t[1]["gold"]]}
    result = {"fixture_sha256": data["fixture_sha256"], "density_counts": {g: sum(len(r["cases"]) for r in data["personas"]
                                                                               if r["density"] == g)
                                                                            for g in ("sparse", "medium", "dense")},
              "scopes": {k: summarize(v) for k, v in scopes.items()}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
