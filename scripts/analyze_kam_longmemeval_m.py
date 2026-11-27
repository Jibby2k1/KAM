#!/usr/bin/env python3
"""Audit long-history session and turn recall at explicitly matched budgets."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

MODES = ("dense", "user", "assistant", "length_corrected", "dense_recent", "asof_dense", "recent_only")
BUDGETS = ("64", "128", "256", "512", "full")
CHECKS = (1, 3, 5, 10, 20)


def evaluate(cases, results):
    totals = {n: {a: Counter() for a in MODES} for n in BUDGETS}
    turn = {name: Counter() for name in ("dense", "user", "length_corrected")}
    complete = {a: [] for a in MODES}
    future_evidence = 0
    for row in cases:
        r = results[row["id"]]
        known = set(row["gold_ids"])
        assert known
        dates, ids = row["dates"], row["source_session_ids"]
        assert len(dates) == len(ids) == r["session_count"]
        assert len(row["author_turn_ids"]) == len(ids)
        available = {sid for sid, date in zip(ids, dates, strict=True) if date <= row["time"]}
        future_evidence += not known <= available
        marked_sessions = {ids[i] for i, t in enumerate(row["author_turn_ids"]) if t}
        assert marked_sessions <= known
        for group in BUDGETS:
            policies = r["full"] if group == "full" else r["nested"][group]
            for arm in MODES:
                slots = policies[arm]
                top = [ids[i] for i in slots]
                assert len(top) == len(set(top)) <= max(CHECKS)
                assert all(0 <= i < len(dates) for i in slots)
                history = totals[group][arm]
                if arm in ("asof_dense", "recent_only"):
                    assert all(dates[i] <= row["time"] for i in slots)
                for k in CHECKS:
                    found = set(top[:k])
                    history["all/" + str(k)] += known <= found
                    history["gold_recalled/" + str(k)] += len(known & found)
                history["one_future_in_top5"] += any(dates[i] > row["time"] for i in slots[:5])
                if group == "full":
                    complete[arm].append(known <= set(top[:5]))
        for arm, picks in r["turns_from_top5"].items():
            assert all(slot in r["full"][arm][:5] for slot, _ in picks)
            assert len(picks) == len(set(map(tuple, picks))) <= 20
            stats = turn[arm]
            for cap in (5, 10, 20):
                found_marked = {ids[slot] for slot, index in picks[:cap] if index in row["author_turn_ids"][slot]}
                stats["any_marked_turn/" + str(cap)] += bool(found_marked)
                stats["marked_turn_each_gold_session/" + str(cap)] += known <= found_marked
    groups = shared_evidence_clusters([{"gold": case["gold_ids"]} for case in cases])
    return {"questions": len(cases), "gold_sessions": sum(len(r["gold_ids"]) for r in cases),
            "incomplete_under_date_cutoff": future_evidence, "shared_evidence_clusters": len(groups),
            "median_distinct_sessions": median([len(set(r["source_session_ids"])) for r in cases]),
            "median_session_occurrences": median([results[r["id"]]["session_count"] for r in cases]),
            "median_chunks": median([results[r["id"]]["chunk_count"] for r in cases]),
            "retrieval": totals, "turn_selection": turn,
            "full_top5_paired_vs_dense": {a: compare(flags, complete["dense"], groups)
                                         for a, flags in complete.items() if a != "dense"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--shards", type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = unpack(a.manifest)
    results = {}
    chunks = sorted(map(unpack, a.shards), key=lambda x: x["shard"])
    assert len(manifest["rows"]) == 500 and [s["shard"] for s in chunks] == list(range(4))
    for s in chunks:
        assert s["fixture_sha256"] == manifest["fixture_sha256"] and s["bank_sha256"] == manifest["session_bank_sha256"]
        assert s["source_sha256"] == manifest["source_sha256"] and s["models"] == chunks[0]["models"]
        assert s["device"] == "NVIDIA L4" and s["shards"] == 4
        assert [r["id"] for r in s["rows"]] == [r["id"] for r in manifest["rows"][s["shard"]::4]]
        results.update({r["id"]: r for r in s["rows"]})
    valid = [q for q in manifest["rows"] if not q["id"].endswith("_abs")]
    assert len(valid) == 470 and all(q["gold_ids"] for q in valid)
    scopes = {"answerable": valid,
              "cutoff_complete": [q for q in valid if set(q["gold_ids"]) <= {
                  sid for sid, date in zip(q["source_session_ids"], q["dates"], strict=True) if date <= q["time"]}]}
    for name in sorted({q["kind"] for q in valid}):
        scopes[name] = [q for q in valid if q["kind"] == name]
    out = {k: manifest[k] for k in ("fixture_sha256", "source_sha256", "small_sha256", "summary")}
    out["measures"] = {k: evaluate(c, results) for k, c in scopes.items()}
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
