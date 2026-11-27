#!/usr/bin/env python3
"""Audit held-out MuSiQue paragraph coverage and shared-evidence clusters."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import quantiles

from analyze_kam_shared_serving import unpack

METHODS = ("bm25", "dense", "hybrid", "one_anchor", "two_anchors")
COHORTS = {2: 224, 3: 192, 4: 160}


def shared_evidence_clusters(cases):
    owners, parent = {}, list(range(len(cases)))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, case in enumerate(cases):
        for doc in case["gold"]:
            j = owners.setdefault(doc, i)
            parent[root(i)] = root(j)
    groups = {}
    for i in range(len(cases)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def compare(new, old, groups):
    diffs = [int(a) - int(b) for a, b in zip(new, old, strict=True)]
    components = [(len(g), sum(diffs[i] for i in g)) for g in groups]
    rng = Random(20371)
    draws = []
    for _ in range(3000):
        pairs = rng.choices(components, k=len(components))
        draws.append(sum(shift for _, shift in pairs) / sum(n for n, _ in pairs))
    bounds = quantiles(draws, n=40)
    return {"delta": round(sum(diffs) / len(diffs), 8), "wins": diffs.count(1), "losses": diffs.count(-1),
            "shared_evidence_cluster_bootstrap95": [round(bounds[0], 8), round(bounds[-1], 8)]}


def evaluate(cases, records, documents):
    totals, hits = {}, {}
    for case in cases:
        row = records[case["id"]]
        assert set(row) == {"id", *METHODS}
        expected = set(case["gold"])
        assert len(expected) == case["hops"] and all(0 <= i < documents for i in expected)
        assert row["one_anchor"][:64] == row["two_anchors"][:64] == row["dense"][:64]
        for mode in METHODS:
            assert len(row[mode]) == len(set(row[mode])) == 128 and all(0 <= i < documents for i in row[mode])
            for depth in (16, 64, 128):
                rank = row[mode][:depth]
                coverage = len(expected.intersection(rank))
                key = f"{mode}/{depth}"
                totals.setdefault(key, Counter()).update({"all_support": coverage == case["hops"],
                                                          "any_support": coverage > 0,
                                                          "supporting_paragraphs": coverage,
                                                          "top1_support": rank[0] in expected})
                hits.setdefault(key, []).append(coverage == case["hops"])
    groups = shared_evidence_clusters(cases)
    contrasts = {f"{a}_vs_{b}": compare(hits[f"{a}/128"], hits[f"{b}/128"], groups)
                 for a, b in (("one_anchor", "dense"), ("two_anchors", "one_anchor"), ("two_anchors", "dense"),
                              ("bm25", "dense"), ("hybrid", "dense"))}
    return {"questions": len(cases), "gold_paragraphs": sum(q["hops"] for q in cases),
            "shared_evidence_clusters": len(groups), "largest_cluster": max(map(len, groups)),
            "retrieval": totals, "paired_support": contrasts}


def summarize(manifest, shards):
    cases = manifest["rows"]
    assert len(cases) == len({q["id"] for q in cases}) == sum(COHORTS.values())
    assert Counter(r["hops"] for r in cases) == COHORTS
    assert len({tuple(r["gold"]) for r in cases}) == len(cases)
    assert set(s["shard"] for s in shards) == set(range(4)) and len(shards) == 4
    for s in shards:
        assert s["fixture_sha256"] == manifest["fixture_sha256"] and s["corpus_sha256"] == manifest["corpus_sha256"]
        assert s["device"] == "NVIDIA L4" and s["shards"] == 4 and s["models"] == shards[0]["models"]
        assert [r["id"] for r in s["rows"]] == [q["id"] for q in cases[s["shard"]::4]]
    rows = {r["id"]: r for s in shards for r in s["rows"]}
    scopes = {"all": cases, **{str(i): [q for q in cases if q["hops"] == i] for i in COHORTS}}
    return {"corpus_sha256": manifest["corpus_sha256"], "fixture_sha256": manifest["fixture_sha256"],
            "documents": manifest["documents"], "sources": manifest["sources"],
            "scopes": {k: evaluate(v, rows, manifest["documents"]) for k, v in scopes.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shards", nargs=4, type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path)
    source.add_argument("--manifest", type=Path)
    p.add_argument("--write-manifest", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.fixture:
        original = a.fixture.read_bytes()
        manifest = json.loads(original)
        manifest["fixture_sha256"] = sha256(original).hexdigest()
        for row in manifest["rows"]:
            del row["question"]
        if a.write_manifest:
            a.write_manifest.write_bytes(gzip.compress(json.dumps(manifest, separators=(",", ":")).encode(), mtime=0))
    else:
        manifest = unpack(a.manifest)
    result = summarize(manifest, [unpack(p) for p in a.shards])
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
