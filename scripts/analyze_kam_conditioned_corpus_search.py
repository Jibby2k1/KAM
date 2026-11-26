#!/usr/bin/env python3
"""Replay paired full-corpus second-hop results from a sealed HotpotQA cohort."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_frozen_sequential_oracle import digest, unpack
from analyze_kam_sentence_verifier import paired

CORPUS_ARMS = ("corpus_unconditioned", "corpus_conditioned")
ARMS = ("broad", "same_pool_sequential", *CORPUS_ARMS)
METRICS = ("support", "joint_exact")
COMPARISONS = (
    ("same_pool_sequential", "broad"),
    ("corpus_conditioned", "broad"),
    ("corpus_conditioned", "same_pool_sequential"),
    ("corpus_conditioned", "corpus_unconditioned"),
)


def verify(shards, fixture_path, pool_path, reference_path):
    fields = ("fixture_sha256", "pools_sha256", "reference_sha256", "index_sha256", "baseline_sha256",
              "config_sha256", "models", "indexed_documents", "candidate_depth", "shards")
    meta = {k: shards[0][k] for k in fields}
    assert all(all(s[k] == v for k, v in meta.items()) for s in shards)
    assert sorted(s["shard"] for s in shards) == list(range(meta["shards"]))
    raw = gzip.decompress(fixture_path.read_bytes())
    sealed = unpack(pool_path)
    assert meta["fixture_sha256"] == sha256(raw).hexdigest() == sealed["fixture_sha256"]
    assert meta["pools_sha256"] == digest(pool_path) and meta["reference_sha256"] == digest(reference_path)
    assert meta["index_sha256"] == sealed["index_sha256"] and meta["candidate_depth"] == 64
    holdout = {r["id"]: set(r["gold_ids"]) for r in json.loads(raw)["rows"]["test"]}
    ref = {r["id"]: r for shard in unpack(reference_path) for r in shard["rows"] if r["split"] == "test"}
    rows = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == len(holdout) == len(ref) == 1024
    assert {r["id"] for r in rows} == holdout.keys() == ref.keys()
    reread = Counter(dict.fromkeys(("broad", "same_pool_sequential"), 0))
    in_pool_disagreements = 0
    near_ties = Counter(dict.fromkeys(CORPUS_ARMS, 0))
    verified = []
    for row in rows:
        id_ = row["id"]
        old = ref[id_]
        ranked = sealed["rows"][id_]
        pool = set(ranked)
        gold = holdout[id_]
        missing = gold - pool
        assert len(pool) == len(ranked) == 64 and len(gold) == 2
        assert old["gold_in_pool"] == len(gold & pool)
        assert row["pool_hash"] == old["pool_hash"] == sha256(json.dumps(ranked, separators=(",", ":")).encode()).hexdigest()
        assert set(row["arms"]) == set(ARMS) and set(row["corpus_hits"]) == set(CORPUS_ARMS)
        first = row["arms"]["same_pool_sequential"]["pair"][0]
        for arm, reference_arm in (("broad", "broad"), ("same_pool_sequential", "sequential")):
            result = row["arms"][arm]
            assert result["pair"] == [ranked[p] for p in old["arms"][reference_arm]["pair"]]
            if result["exact"] != old["arms"][reference_arm]["exact"]:
                reread[arm] += 1
        for arm in CORPUS_ARMS:
            hits = row["corpus_hits"][arm]
            assert len(hits) == len(set(hits)) == meta["candidate_depth"] and first not in hits
            assert row["arms"][arm]["pair"] == [first, hits[0]]
            near_ties[arm] += row["top1_margin"][arm] <= 1e-7
        global_second = row["arms"]["corpus_conditioned"]["pair"][1]
        local_second = row["arms"]["same_pool_sequential"]["pair"][1]
        in_pool_disagreements += global_second in pool and global_second != local_second
        seen = {}
        for result in row["arms"].values():
            pair = tuple(result["pair"])
            assert len(pair) == len(set(pair)) == 2
            assert pair not in seen or seen[pair] == result["exact"]
            seen[pair] = result["exact"]
            result["support"] = set(pair) == gold
            result["joint_exact"] = result["support"] and result["exact"]
        verified.append(row | {"pool": pool, "gold": gold, "missing": missing, "first": first,
                               "gold_linked": old["gold_linked"]})
    parity = {"reread_differences_vs_reference": dict(reread),
              "conditioned_top1_inside_pool_but_differs_from_same_pool": in_pool_disagreements,
              "corpus_top1_near_ties": dict(near_ties)}
    return meta, verified, parity


def panel(rows):
    outcomes = {arm: {m: [r["arms"][arm][m] for r in rows] for m in METRICS} for arm in ARMS}
    correct = {arm: {m: sum(values) for m, values in cols.items()} for arm, cols in outcomes.items()}
    compared = {
        arm + "_vs_" + base: {m: paired(outcomes[arm][m], outcomes[base][m]) for m in METRICS}
        for arm, base in COMPARISONS
    } if rows else {}
    expansion = {}
    for arm in CORPUS_ARMS:
        diagnostic = Counter(dict.fromkeys(("top1_outside_pool", "top1_missing_gold", "top64_any_missing_gold",
                                            "first_plus_top64_both_gold", "first_plus_top64_union_pool_both_gold"), 0))
        for r in rows:
            hits = r["corpus_hits"][arm]
            candidates = {r["first"], *hits}
            diagnostic["top1_outside_pool"] += hits[0] not in r["pool"]
            diagnostic["top1_missing_gold"] += hits[0] in r["missing"]
            diagnostic["top64_any_missing_gold"] += bool(r["missing"].intersection(hits))
            diagnostic["first_plus_top64_both_gold"] += r["first"] in r["gold"] and r["gold"] <= candidates
            diagnostic["first_plus_top64_union_pool_both_gold"] += (
                r["first"] in r["gold"] and r["gold"] <= candidates | r["pool"]
            )
        expansion[arm] = dict(diagnostic)
    oracle_arms = ("broad", "same_pool_sequential", "corpus_conditioned")
    oracle = {m: sum(any(r["arms"][a][m] for a in oracle_arms) for r in rows) for m in METRICS}
    return {"n": len(rows), "first_hop_gold": sum(r["first"] in r["gold"] for r in rows),
            "answer_exact": {a: sum(r["arms"][a]["exact"] for r in rows) for a in ARMS},
            "correct": correct, "paired": compared, "corpus_expansion": expansion,
            "oracle_broad_same_pool_conditioned": oracle}


def summarize(shards, fixture, pools, reference):
    meta, rows, parity = verify(shards, fixture, pools, reference)
    missing = [r for r in rows if r["missing"]]
    one = [r for r in missing if len(r["missing"]) == 1]
    cases = {
        "all": rows,
        "both_in_frozen_pool": [r for r in rows if not r["missing"]],
        "either_absent_from_frozen_pool": missing,
        "one_absent_from_frozen_pool": one,
        "one_absent_first_hop_gold": [r for r in one if r["first"] in r["gold"]],
        "one_absent_first_hop_gold_both_in_index": [r for r in one if r["first"] in r["gold"] and None not in r["gold"]],
        "one_absent_first_hop_wrong": [r for r in one if r["first"] not in r["gold"]],
        "both_absent_from_frozen_pool": [r for r in missing if len(r["missing"]) == 2],
        "gold_absent_from_index": [r for r in missing if None in r["gold"]],
        "gold_unlinked_in_frozen_pool": [r for r in rows if r["gold_linked"] is False],
    }
    return {**meta, **parity, "gpus": dict(sorted(Counter(s["gpu"] for s in shards).items())),
            "cohorts": {k: panel(rs) for k, rs in cases.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    for name in ("fixture", "pools", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--write-replay", type=Path)
    args = parser.parse_args()
    shards = [unpack(p) for p in sorted(args.input.glob("eval-*.json.gz"))] if args.input.is_dir() else unpack(args.input)
    output = summarize(shards, args.fixture, args.pools, args.reference)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if args.write_replay:
        args.write_replay.write_bytes(gzip.compress(json.dumps(shards, separators=(",", ":"), allow_nan=False).encode(),
                                                   mtime=0))


if __name__ == "__main__":
    main()
