#!/usr/bin/env python3
"""Audit reranking over sealed pools and exact corpus unions, then replay paired outcomes."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_frozen_sequential_oracle import digest, unpack
from analyze_kam_sentence_verifier import paired

ARMS = ("broad", "unconditioned_union", "conditioned_union")
METRICS = ("support", "joint_exact")
COMPARISONS = (("conditioned_union", "broad"), ("unconditioned_union", "broad"),
               ("conditioned_union", "unconditioned_union"))


def audit(shards, fixture_path, pool_path, fit_path):
    fields = ("fixture_sha256", "pools_sha256", "index_sha256", "config_sha256", "fitted_sha256",
              "heads_sha256", "ablations_sha256", "baseline_sha256", "models", "indexed_documents", "shards")
    meta = {k: shards[0][k] for k in fields}
    assert all(all(s[k] == v for k, v in meta.items()) for s in shards)
    assert sorted(s["shard"] for s in shards) == list(range(meta["shards"]))
    raw = gzip.decompress(fixture_path.read_bytes())
    sealed = unpack(pool_path)
    assert meta["fixture_sha256"] == sha256(raw).hexdigest() == sealed["fixture_sha256"]
    assert meta["index_sha256"] == sealed["index_sha256"]
    assert meta["pools_sha256"] == digest(pool_path) and meta["fitted_sha256"] == digest(fit_path)
    fixture = json.loads(raw)["rows"]["test"]
    assert len({t for q in fixture for t in q["gold_titles"]}) == 2 * len(fixture)
    gold_by_id = {q["id"]: set(q["gold_ids"]) for q in fixture}
    rows = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == len(gold_by_id) == len(sealed["rows"]) == 1024
    assert {r["id"] for r in rows} == gold_by_id.keys() == sealed["rows"].keys()

    verified = []
    for r in rows:
        id_ = r["id"]
        original = sealed["rows"][id_]
        pool, gold = set(original), gold_by_id[id_]
        assert len(pool) == len(original) == 64 and len(gold) == 2
        assert r["first"] in pool and r["pool_hash"] == sha256(
            json.dumps(original, separators=(",", ":")).encode()).hexdigest()
        assert set(r["arms"]) == set(ARMS) and set(r["corpus_hits"]) == set(ARMS[1:])
        seen, arms = {}, {}
        for arm in ARMS:
            top = r["corpus_hits"].get(arm, [])
            if arm != "broad":
                assert len(top) == len(set(top)) == 64 and r["first"] not in top
            candidates = pool | set(top)
            result = r["arms"][arm]
            pair = tuple(result["pair"])
            assert len(pair) == len(set(pair)) == 2 and set(pair) <= candidates
            assert pair not in seen or seen[pair] == result["exact"]
            seen[pair] = result["exact"]
            covered = gold <= candidates
            support = set(pair) == gold
            assert support <= result["gold_pair_in_shortlist"] <= covered
            arms[arm] = result | {"covered": covered, "support": support, "joint_exact": support and result["exact"],
                                  "candidate_count": len(candidates), "uses_added": not set(pair) <= pool}
        verified.append({"id": id_, "missing": len(gold - pool), "first_gold": r["first"] in gold, "arms": arms})
    return verified, meta


def panel(rows):
    outcomes = {arm: {m: [r["arms"][arm][m] for r in rows] for m in METRICS} for arm in ARMS}
    compared = {arm + "_vs_" + base: {m: paired(outcomes[arm][m], outcomes[base][m]) for m in METRICS}
                for arm, base in COMPARISONS} if rows else {}
    stages = {}
    for arm in ARMS:
        values = [r["arms"][arm] for r in rows]
        stages[arm] = {k: sum(v[k] for v in values) for k in ("covered", "gold_pair_in_shortlist", "support",
                                                            "joint_exact", "uses_added")}
        if values:
            counts = [v["candidate_count"] for v in values]
            stages[arm]["candidate_count"] = {"min": min(counts), "max": max(counts),
                                               "mean": round(sum(counts) / len(counts), 5)}
    losses = {}
    for arm in ARMS[1:]:
        support_losses = [r["arms"][arm] for r in rows
                          if r["arms"]["broad"]["support"] and not r["arms"][arm]["support"]]
        losses[arm] = {"broad_support_lost": len(support_losses),
                       "broad_support_lost_with_added_document": sum(r["uses_added"] for r in support_losses),
                       "broad_support_lost_gold_pair_left_shortlist": sum(not r["gold_pair_in_shortlist"]
                                                                        for r in support_losses)}
    return {"n": len(rows), "stages": stages, "paired": compared,
            "loss_diagnostics": losses,
            "broad_union_oracle": {m: sum(a or b for a, b in zip(outcomes["broad"][m],
                                                               outcomes["conditioned_union"][m], strict=True))
                                  for m in METRICS}}


def summarize(shards, fixture, pools, fitted):
    rows, meta = audit(shards, fixture, pools, fitted)
    cohorts = {
        "all": rows,
        "complete_original": [r for r in rows if r["missing"] == 0],
        "incomplete_original": [r for r in rows if r["missing"] > 0],
        "one_missing": [r for r in rows if r["missing"] == 1],
        "two_missing": [r for r in rows if r["missing"] == 2],
        "incomplete_first_gold": [r for r in rows if r["missing"] > 0 and r["first_gold"]],
        "incomplete_first_wrong": [r for r in rows if r["missing"] > 0 and not r["first_gold"]],
    }
    return {**meta, "gpus": dict(sorted(Counter(s["gpu"] for s in shards).items())),
            "cohorts": {name: panel(cases) for name, cases in cohorts.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    for key in ("fixture", "pools", "fitted", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--write-replay", type=Path)
    a = parser.parse_args()
    shards = [unpack(p) for p in sorted(a.input.glob("score-*.json.gz"))] if a.input.is_dir() else unpack(a.input)
    result = summarize(shards, a.fixture, a.pools, a.fitted)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        data = json.dumps(shards, separators=(",", ":"), allow_nan=False).encode()
        a.write_replay.write_bytes(gzip.compress(data, mtime=0))


if __name__ == "__main__":
    main()
