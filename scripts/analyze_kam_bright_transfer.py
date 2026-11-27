#!/usr/bin/env python3
"""Evaluate frozen BRIGHT retrieval lists at equal output depth."""

import argparse
from collections import Counter
import gzip
import json
from math import log2
from pathlib import Path

import numpy as np

DEPTH = 64


def group_queries(rows):
    groups = [{i} for i in range(len(rows))]
    seen = {}
    for i, row in enumerate(rows):
        for other in row["gold_docs"]:
            previous = seen.setdefault(other, i)
            first, second = groups[previous], groups[i]
            if first is second:
                continue
            union = first | second
            for member in union:
                groups[member] = union
    return sorted({tuple(sorted(g)) for g in groups})


def metrics(rank, gold, count):
    hits = [i for i, doc in enumerate(rank, 1) if doc in gold]
    dcg = sum(1 / log2(i + 1) for i in hits if i <= 10)
    normalizer = sum(1 / log2(i + 1) for i in range(1, min(10, count) + 1))
    return dcg / normalizer, len(hits) / count, float(len(hits) == count)


def resample(groups, draws=5000):
    rng = np.random.default_rng(829548)
    count = len(groups)
    return rng.integers(0, count, size=(draws, count))


def limits(values, groups, sample):
    mass = np.asarray([sum(values[i] for i in g) for g in groups])
    counts = np.asarray(list(map(len, groups)))
    estimates = mass[sample].sum(axis=1) / counts[sample].sum(axis=1)
    return np.quantile(estimates, [.025, .975]).tolist()


def analyze(manifest, ranked):
    assert all(manifest[k] == ranked[k] for k in ("fixture_sha256", "task", "documents"))
    n = manifest["queries"]
    assert len(manifest["rows"]) == len(ranked["rows"]) == n
    by_policy, valid, searches = {}, Counter(), Counter()
    groups = group_queries(manifest["rows"])
    for label, evaluated in zip(manifest["rows"], ranked["rows"], strict=True):
        assert label["id"] == evaluated["id"]
        gold, banned = set(label["gold"]), set(label["excluded"])
        total = len(label["gold_docs"])
        assert total > 0 and len(gold) + label["unindexed_gold"] == total
        labels = set(evaluated["valid_plans"])
        assert labels == set(evaluated["plan_searches"])
        alternatives = {"hybrid_windows", "hybrid_anchor", *("hybrid_plan/" + k for k in labels)} if ranked.get(
            "hybrid_augmentation") else set()
        assert set(evaluated["ranks"]) == labels | {"bm25", "dense", "one_pass_windows", "conditioned", "hybrid"} | alternatives
        for label, accepted in evaluated["valid_plans"].items():
            steps = evaluated["plan_searches"][label]
            assert steps in ((2, 3, 4) if accepted else (0,))
            valid[label] += accepted
            searches[label] += steps
        for name, rank in evaluated["ranks"].items():
            assert len(rank) == len(set(rank)) == DEPTH and set(rank).isdisjoint(banned)
            assert all(0 <= i < ranked["documents"] for i in rank)
            if name in labels and not evaluated["valid_plans"][name]:
                assert rank == evaluated["ranks"]["dense"]
                if alternatives:
                    assert evaluated["ranks"]["hybrid_plan/" + name] == evaluated["ranks"]["hybrid"]
            by_policy.setdefault(name, []).append(metrics(rank, gold, total))
    sample = resample(groups)
    fields = ("ndcg10", "recall64", "complete64")
    by_policy = {k: np.asarray(v) for k, v in by_policy.items()}
    result = {}
    for method, values in by_policy.items():
        assert len(values) == n
        record = {"mean": dict(zip(fields, map(float, values.mean(axis=0)), strict=True)),
                  "complete_questions": int(sum(values[:, 2])), "vs_baseline": {}}
        for baseline in ("dense", "bm25", "hybrid"):
            delta = values - by_policy[baseline]
            record["vs_baseline"][baseline] = {
                k: {"delta": float(v.mean()), "cluster_interval95": limits(v, groups, sample)}
                for k, v in zip(fields, delta.T, strict=True)}
        if method in valid:
            record.update(valid_plans=valid[method], additional_searches=searches[method])
        if method.startswith("hybrid_plan/"):
            source = method.removeprefix("hybrid_plan/")
            record.update(valid_plans=valid[source], additional_searches=searches[source])
        result[method] = record
    return {"fixture_sha256": ranked["fixture_sha256"], "task": ranked["task"],
            "documents": ranked["documents"], "queries": n, "shared_gold_clusters": len(groups),
            "largest_cluster": max(map(len, groups)), "device": ranked["device"],
            "revisions": ranked["revisions"], "weight_records": ranked["weight_records"],
            "index_gpu_seconds": ranked["index_gpu_seconds"], "bm25_build_seconds": ranked["bm25_build_seconds"],
            "methods": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--ranks", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    ranked = json.loads(gzip.decompress(args.ranks.read_bytes()))
    result = analyze(manifest, ranked)
    if args.reference:
        prior = json.loads(gzip.decompress(args.reference.read_bytes()))
        assert all(prior[k] == ranked[k] for k in ("fixture_sha256", "task", "revisions", "documents"))
        for old, new in zip(prior["rows"], ranked["rows"], strict=True):
            assert all(new[k] == old[k] for k in ("id", "valid_plans", "plan_searches"))
            assert all(new["ranks"][k] == value for k, value in old["ranks"].items())
        result["exact_reference_records"] = len(prior["rows"])
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
