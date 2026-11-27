#!/usr/bin/env python3
"""Compare head-only and expanded document embeddings at the same BRIGHT depth."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from analyze_kam_bright_transfer import DEPTH, group_queries, limits, metrics, resample

ARMS = ("bm25", "dense_head", "dense_tail", "dense_windows", "hybrid_head", "hybrid_tail", "hybrid_windows")
METRICS = ("ndcg10", "recall64", "complete64")


def rankings(cases, data, original, lengths, cap, policies):
    n_docs = len(lengths)
    scores = {a: [] for a in policies}
    long_recall = {a: [] for a in policies}
    long_positions = []
    for i, (q, row, baseline) in enumerate(zip(cases, data, original, strict=True)):
        assert q["id"] == row["id"]
        if baseline:
            assert row["id"] == baseline["id"]
            old_ranks = baseline["ranks"]
            if "dense_head" not in old_ranks:
                old_ranks = {m: old_ranks[k] for m, k in (("bm25", "bm25"), ("dense_head", "dense"), ("hybrid_head", "hybrid"))}
            assert all(row["ranks"][k] == v for k, v in old_ranks.items())
        assert set(row["ranks"]) == set(policies)
        expected, excluded = set(q["gold"]), set(q["excluded"])
        assert not q["unindexed_gold"]
        long = {d for d in expected if lengths[d] > cap}
        if long:
            long_positions.append(i)
        for name, checked in row["ranks"].items():
            ids = set(checked)
            assert ids.isdisjoint(excluded) and len(ids) == len(checked) == DEPTH
            assert min(ids) >= 0 and max(ids) < n_docs
            scores[name].append(metrics(checked, expected, len(expected)))
            if long:
                long_recall[name].append(len(ids & long) / len(long))
    return scores, long_recall, long_positions


def evaluate(fixture, measured, former):
    identifiers = ("fixture_sha256", "task", "documents")
    assert all(measured[k] == fixture[k] for k in identifiers)
    if former:
        assert all(former[k] == fixture[k] for k in identifiers)
    docs, cutoff = measured["document_token_counts"], measured["window_tokens"]
    assert len(docs) == fixture["documents"]
    assert sum(x > cutoff for x in docs) == measured["documents_with_tail"]
    shams = measured.get("sham_moved_documents", {})
    assert all(0 < n <= measured["documents_with_tail"] for n in shams.values())
    policies = ARMS + tuple(f"{m}_sham/{k}" for m in ("dense", "hybrid") for k in sorted(shams))
    # Only fully indexed gold sets are admitted to this study. Local integer
    # document IDs reproduce the source-ID grouping without publishing names.
    all_queries = [q | {"gold_docs": q["gold"]} for q in fixture["rows"]]
    original = former["rows"] if former else [None] * len(all_queries)
    score, long_score, long_items = rankings(all_queries, measured["rows"], original, docs, cutoff, policies)
    groups = group_queries(all_queries)
    selected_groups = group_queries([all_queries[i] for i in long_items])
    complete, long_only = resample(groups), resample(selected_groups)
    result = {}
    for name in policies:
        old = "hybrid_head" if name.startswith("hybrid") or name == "bm25" else "dense_head"
        main = np.asarray(score[name])
        extra = np.asarray(long_score[name])
        delta = main - np.asarray(score[old])
        long_delta = extra - np.asarray(long_score[old])
        result[name] = {"reference": old, "scores": dict(zip(METRICS, main.mean(axis=0).tolist(), strict=True)),
                        "delta": {k: {"mean": float(v.mean()), "cluster_interval95": limits(v, groups, complete)}
                                  for k, v in zip(METRICS, delta.T, strict=True)},
                        "long_doc_recall64": float(extra.mean()),
                        "long_doc_delta": {"mean": float(long_delta.mean()),
                                           "cluster_interval95": limits(long_delta, selected_groups, long_only)}}
        if "_sham/" in name:
            actual = name.split("_", 1)[0] + "_windows"
            difference = np.asarray(score[actual]) - main
            result[name]["correct_content_minus_sham"] = {
                k: {"mean": float(v.mean()), "cluster_interval95": limits(v, groups, complete)}
                for k, v in zip(METRICS, difference.T, strict=True)}
    metadata = {k: v for k, v in measured.items() if k not in ("document_token_counts", "rows")}
    return metadata | {"questions": len(all_queries), "archive_rank_matches": len(original) if former else 0,
                       "questions_with_long_gold": len(long_items),
                       "gold_with_truncated_head": sum(docs[d] > cutoff for q in all_queries for d in q["gold"]),
                       "gold_records": sum(len(q["gold"]) for q in all_queries),
                       "shared_evidence_clusters": len(groups), "arms": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("manifest", "measurement", "reference", "output"):
        p.add_argument("--" + k, type=Path, required=(k != "reference"))
    a = p.parse_args()
    meta = json.loads(a.manifest.read_bytes())
    first, second = [json.loads(gzip.decompress(x.read_bytes())) if x else None for x in (a.measurement, a.reference)]
    a.output.write_text(json.dumps(evaluate(meta, first, second), sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
