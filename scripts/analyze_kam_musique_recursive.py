#!/usr/bin/env python3
"""Compare full-chain coverage on matching frozen MuSiQue candidate pools."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_native_musique import COHORTS, METHODS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

ADDITIONAL = ("parallel_three", "recursive_three", "annotated_only", "resolved_answers")


def evaluate(cases, records, original):
    metrics, flags = {}, {}
    for arm in (*METHODS, *ADDITIONAL):
        totals, found = Counter(), []
        for q in cases:
            id_ = q["id"]
            baseline, new = original[id_], records[id_]
            gold = set(q["gold"])
            rank = (baseline if arm in METHODS else new)[arm]
            n = len(gold.intersection(rank))
            assert len(rank) == len(set(rank)) == 128
            if arm in ADDITIONAL:
                assert rank[:64] == baseline["dense"][:64]
            totals.update({"all_support": n == q["hops"], "supporting_paragraphs": n,
                           "any_support": n > 0})
            found.append(n == q["hops"])
        flags[arm], metrics[arm] = found, totals
    groups = shared_evidence_clusters(cases)
    comparisons = [("recursive_three", "parallel_three"), ("recursive_three", "hybrid"),
                   ("parallel_three", "two_anchors"), ("annotated_only", "hybrid"),
                   ("resolved_answers", "annotated_only"), ("resolved_answers", "hybrid")]
    hop_targets = {}
    for arm in ADDITIONAL[2:]:
        key = arm + "_target_ranks"
        assert all(len(records[q["id"]][key]) == q["hops"] for q in cases)
        counts = []
        for i in range(max(q["hops"] for q in cases)):
            values = [records[q["id"]][key][i] for q in cases if q["hops"] > i]
            counts.append({"questions": len(values), "top_1": values.count(1),
                           "top_16": sum(x is not None and x <= 16 for x in values),
                           "not_top_128": values.count(None)})
        hop_targets[arm] = counts
    return {"questions": len(cases), "metrics": metrics, "annotated_target_rank": hop_targets,
            "paired": {f"{a}_vs_{b}": compare(flags[a], flags[b], groups) for a, b in comparisons}}


def run(manifest, previous, recursive):
    assert len(previous) == len(recursive) == 4
    assert {s["shard"] for s in recursive} == set(range(4))
    docs = manifest["documents"]
    original, records = {}, {}
    for previous_hash, old in previous.items():
        new = next(s for s in recursive if s["baseline_sha256"] == previous_hash)
        assert new["fixture_sha256"] == old["fixture_sha256"] == manifest["fixture_sha256"]
        assert new["corpus_sha256"] == old["corpus_sha256"] == manifest["corpus_sha256"]
        assert new["models"] == old["models"] and new["device"] == "NVIDIA L4"
        assert (new["shard"], new["shards"]) == (old["shard"], old["shards"])
        assert [r["id"] for r in new["rows"]] == [r["id"] for r in old["rows"]]
        for row in new["rows"]:
            assert all(0 <= i < docs for arm in ADDITIONAL for i in row[arm])
        original.update({r["id"]: r for r in old["rows"]})
        records.update({r["id"]: r for r in new["rows"]})
    assert len({s["oracle_sha256"] for s in recursive}) == 1
    cases = manifest["rows"]
    assert len(cases) == len(records) == sum(COHORTS.values())
    arms = {"all": cases, **{str(i): [q for q in cases if q["hops"] == i] for i in COHORTS}}
    return {"oracle_sha256": recursive[0]["oracle_sha256"], "fixture_sha256": manifest["fixture_sha256"],
            "corpus_sha256": manifest["corpus_sha256"], "scopes": {s: evaluate(q, records, original)
                                                                    for s, q in arms.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", nargs=4, required=True, type=Path)
    p.add_argument("--recursive", nargs=4, required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    previous = {sha256(path.read_bytes()).hexdigest(): unpack(path) for path in a.baseline}
    result = run(unpack(a.manifest), previous, [unpack(p) for p in a.recursive])
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
