#!/usr/bin/env python3
"""Audit the effect of correcting one intermediate answer in MuSiQue."""

import argparse
import json
from pathlib import Path

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

POLICIES = ("top1", "gold_at_1", "gold_at_2", "gold_at_3")
METRICS = ("support", "final_subanswer", "joint", "later_top1")


def assess(cases, done, original, policies):
    measures = {arm: {metric: [] for metric in METRICS} for arm in policies}
    for i, name in enumerate(policies):
        for q in cases:
            gold = set(q["gold"])
            row = done[q["id"]][name]
            support = gold <= set(row["pool"])
            final = bool(row["answer_em"][-1])
            later_top1 = row["target_ranks"][i] == 1
            for metric, hit in zip(METRICS, (support, final, support and final, later_top1), strict=True):
                measures[name][metric].append(hit)
    groups = shared_evidence_clusters(cases)
    pairs = {}
    for i, arm in enumerate(policies[1:], 1):
        baseline = measures["top1"] | {
            "later_top1": [original[q["id"]]["target_ranks"][i] == 1 for q in cases]}
        pairs[arm] = {tag: compare(got, baseline[tag], groups) for tag, got in measures[arm].items()}
    counts = {arm: {tag: sum(v) for tag, v in values.items() if tag != "later_top1"}
              for arm, values in measures.items()}
    next_hop = {arm: {"oracle": sum(measures[arm]["later_top1"]),
                      "top1": sum(original[q["id"]]["target_ranks"][i] == 1 for q in cases)}
                for i, arm in enumerate(policies[1:], 1)}
    return {"questions": len(cases), "clusters": len(groups), "counts": counts,
            "next_hop_rank_one": next_hop, "paired": pairs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--baseline", required=True, type=Path, nargs=4)
    p.add_argument("--measurement", required=True, type=Path, nargs=4)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    source = unpack(a.manifest)
    cases = source["rows"]
    old, new = [sorted(map(unpack, files), key=lambda r: r["shard"]) for files in (a.baseline, a.measurement)]
    done, original = {}, {}
    fields = ("fixture_sha256", "corpus_sha256", "models", "templates_sha256", "scoring_labels_sha256", "baseline_sha256")
    for i, (base, measured) in enumerate(zip(old, new, strict=True)):
        assert (base["shard"], measured["shard"], base["shards"], measured["shards"]) == (i, i, 4, 4)
        assert base["device"] == measured["device"] == "NVIDIA L4"
        assert all(base[key] == measured[key] for key in fields)
        assert measured["fixture_sha256"] == source["fixture_sha256"] and measured["corpus_sha256"] == source["corpus_sha256"]
        for q, before, after in zip(cases[i::4], base["rows"], measured["rows"], strict=True):
            id_, hops = q["id"], q["hops"]
            assert id_ == before["id"] == after["id"] and id_ not in done
            original[id_] = before["arms"]["top1"]
            done[id_] = after["arms"]
            assert done[id_]["top1"] == original[id_]
            assert set(done[id_]) == set(POLICIES[:hops])
            for step, mode in enumerate(POLICIES[1:hops], 1):
                row = done[id_][mode]
                assert len(row["pool"]) == len(set(row["pool"])) == 128
                assert row["pool"][:64] == original[id_]["pool"][:64]
                for key in ("target_ranks", "answer_em", "reader_context_ids", "gold_in_reader"):
                    assert len(row[key]) == hops and row[key][:step] == original[id_][key][:step]
    assert len(done) == len(cases)
    scopes = {str(i): assess([q for q in cases if q["hops"] == i], done, original, POLICIES[:i]) for i in (2, 3, 4)}
    result = {"fixture_sha256": source["fixture_sha256"], "exact_top1_replays": len(cases), "depths": scopes}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
