#!/usr/bin/env python3
"""Audit frozen conditioned search rank, support coverage, and scan times."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles

from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

ARMS = ("exact", "full_rescore", "probe_16", "probe_64", "probe_256")


def timing(values):
    q = quantiles(values, n=100, method="inclusive")
    return {"p50_ms": round(median(values) * 1000, 5), "p95_ms": round(q[94] * 1000, 5)}


def coverage(cases, original, archive, observed):
    methods = {}
    exact = [(set(q["gold_ids"]) <= set(original[q["id"]]).union(observed[q["id"]]["methods"]["exact"]["ranks"]))
             for q in cases]
    for arm in ARMS:
        covered, archive_recall, same_set, same_order = [], [], [], []
        timed, scanned = [], []
        case_drifts = []
        for q in cases:
            id_ = q["id"]
            record = observed[id_]["methods"][arm]
            ids = record["ranks"]
            assert len(ids) == len(set(ids)) == 64
            assert archive[id_]["first"] not in ids
            expected = archive[id_]["corpus_hits"]["conditioned_union"]
            reference = observed[id_]["methods"]["exact"]["ranks"]
            covered.append(set(q["gold_ids"]) <= set(original[id_]).union(ids))
            archive_recall.append(len(set(expected).intersection(ids)) / len(expected))
            same_set.append(set(ids) == set(reference))
            same_order.append(ids == reference)
            timed.append(record["seconds"])
            scanned.append(record["scanned"])
            case_drifts.append(len(set(ids).intersection(reference)))
        methods[arm] = {"complete": sum(covered), "paired_complete_vs_exact": paired(covered, exact),
                        "identical_sets_vs_exact": sum(same_set), "identical_order_vs_exact": sum(same_order),
                        "mean_overlap_vs_exact": sum(case_drifts) / len(case_drifts) / 64,
                        "mean_overlap_vs_archive": sum(archive_recall) / len(archive_recall),
                        "scanned_median": median(scanned), "scanned_max": max(scanned), **timing(timed)}
    return {"queries": len(cases), "metrics": methods}


def analyze(args, files):
    chunks = sorted(map(unpack, files), key=lambda x: x["shard"])
    assert [x["shard"] for x in chunks] == list(range(4))
    fixture = gzip.decompress(args.fixture.read_bytes())
    cases = json.loads(fixture)["rows"]["test"]
    inputs = unpack(args.pools)
    prior = unpack(args.reference)
    index_hash = inputs["index_sha256"]
    fixture_hash = sha256(fixture).hexdigest()
    assert all(c["fixture_sha256"] == fixture_hash == inputs["fixture_sha256"] and c["index_sha256"] == index_hash
               and c["pools_sha256"] == sha256(args.pools.read_bytes()).hexdigest()
               and c["archive_sha256"] == sha256(args.reference.read_bytes()).hexdigest() for c in chunks)
    meta = chunks[0]
    assert all(c["partition_sha256"] == meta["partition_sha256"] and c["device"] == meta["device"]
               and c["indexed"] == meta["indexed"] for c in chunks)
    archive = {q["id"]: q for c in prior for q in c["rows"]}
    observed, original = {}, inputs["rows"]
    for chunk in chunks:
        assert {q["id"] for q in chunk["rows"]} == {r["id"] for r in cases[chunk["shard"]::4]}
        observed.update({q["id"]: q for q in chunk["rows"]})
    incomplete = [q for q in cases if not set(q["gold_ids"]) <= set(original[q["id"]])]
    sub = {"all": cases, "incomplete": incomplete,
           "incomplete_with_gold_anchor": [q for q in incomplete if archive[q["id"]]["first"] in q["gold_ids"]],
           "incomplete_without_gold_anchor": [q for q in incomplete if archive[q["id"]]["first"] not in q["gold_ids"]]}
    result = {k: meta[k] for k in ("device", "indexed", "partition_sha256", "coarse_index_gpu_bytes",
                                 "full_half_bytes", "full_exact_bytes", "index_sha256", "fixture_sha256")}
    result["scopes"] = {name: coverage(items, original, archive, observed) for name, items in sub.items()}
    return result


def audit_fused(args):
    compared = analyze(args, args.fused)
    chunks = list(map(unpack, args.fused))
    rows = [r["methods"] for c in chunks for r in c["rows"]]
    matched = {}
    for n in (64, 256):
        gathered, tiled = (f"{arm}_{n}" for arm in ("probe", "fused"))
        assert all(a[gathered]["scanned"] == a[tiled]["scanned"] for a in rows)
        paired_ms = [(a[tiled]["seconds"] - a[gathered]["seconds"]) * 1000 for a in rows]
        matched[str(n)] = {"n_identical_orders": sum(a[gathered]["ranks"] == a[tiled]["ranks"] for a in rows),
                           "n_identical_sets": sum(set(a[gathered]["ranks"]) == set(a[tiled]["ranks"]) for a in rows),
                           "gathered": timing([a[gathered]["seconds"] for a in rows]),
                           "fused": timing([a[tiled]["seconds"] for a in rows]),
                           "p50_paired_fused_minus_gathered_ms": median(paired_ms)}
    return {"partition_sha256": compared["partition_sha256"], "device": compared["device"],
            "queries": len(rows), "probes": matched}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "pools", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--l4", type=Path, nargs=4, required=True)
    p.add_argument("--b200", type=Path, nargs=4)
    p.add_argument("--fused", type=Path, nargs=4)
    a = p.parse_args()
    result = {"l4": analyze(a, a.l4)}
    if a.b200:
        result["b200"] = analyze(a, a.b200)
        assert result["l4"]["partition_sha256"] == result["b200"]["partition_sha256"]
    if a.fused:
        result["fused"] = audit_fused(a)
        assert result["l4"]["partition_sha256"] == result["fused"]["partition_sha256"]
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
