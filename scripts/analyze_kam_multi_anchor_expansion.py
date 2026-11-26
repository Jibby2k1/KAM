#!/usr/bin/env python3
"""Audit the fixed augmentation budget and replay evidence coverage by anchor count."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_frozen_sequential_oracle import digest, unpack
from analyze_kam_sentence_verifier import paired

ARMS = ("unconditioned", "anchors_1", "anchors_2", "anchors_4")


def verify(shards, fixture_path, pool_path):
    fields = ("fixture_sha256", "pools_sha256", "index_sha256", "config_sha256", "baseline_sha256",
              "models", "indexed_documents", "shards")
    meta = {key: shards[0][key] for key in fields}
    assert all(all(s[key] == value for key, value in meta.items()) for s in shards)
    assert sorted(s["shard"] for s in shards) == list(range(meta["shards"]))
    raw = gzip.decompress(fixture_path.read_bytes())
    pools = unpack(pool_path)
    assert meta["fixture_sha256"] == sha256(raw).hexdigest() == pools["fixture_sha256"]
    assert meta["pools_sha256"] == digest(pool_path) and meta["index_sha256"] == pools["index_sha256"]
    fixture = json.loads(raw)["rows"]["test"]
    gold = {q["id"]: set(q["gold_ids"]) for q in fixture}
    assert len({t for q in fixture for t in q["gold_titles"]}) == len(fixture) * 2
    rows = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == len(gold) == len(pools["rows"]) == 1024
    assert {r["id"] for r in rows} == gold.keys() == pools["rows"].keys()
    output = []
    for r in rows:
        id_ = r["id"]
        original = pools["rows"][id_]
        present = set(original)
        assert r["pool_hash"] == sha256(json.dumps(original, separators=(",", ":")).encode()).hexdigest()
        assert len(present) == len(original) == 64 and len(gold[id_]) == 2
        anchor = r["anchors"]
        assert len(set(anchor)) == len(anchor) == 4 and set(anchor) <= present
        missing = gold[id_] - present
        assert set(r["added"]) == set(ARMS)
        outcomes = {}
        for arm, ranked in r["added"].items():
            added = set(ranked)
            assert len(added) == len(ranked) == 64 and added.isdisjoint(present)
            outcomes[arm] = {"both": missing <= added, "any_missing": bool(missing & added)}
        output.append({"missing": len(missing), "anchor_gold": [doc in gold[id_] for doc in anchor], "arms": outcomes})
    return output, meta


def panel(rows):
    choices = {a: [r["arms"][a]["both"] for r in rows] for a in ARMS}
    contrasts = (("anchors_1", "unconditioned"), ("anchors_2", "anchors_1"), ("anchors_4", "anchors_1"))
    return {"n": len(rows), "original_both": sum(r["missing"] == 0 for r in rows),
            "anchor_gold": {str(k): sum(any(r["anchor_gold"][:k]) for r in rows) for k in (1, 2, 4)},
            "both_supports_in_union": {a: sum(xs) for a, xs in choices.items()},
            "any_missing_found": {a: sum(r["arms"][a]["any_missing"] for r in rows) for a in ARMS},
            "paired_coverage": {a + "_vs_" + b: paired(choices[a], choices[b]) for a, b in contrasts} if rows else {}}


def summarize(shards, fixture, pools):
    rows, meta = verify(shards, fixture, pools)
    missing = [r for r in rows if r["missing"]]
    groups = {"all": rows, "incomplete": missing,
              "one_missing": [r for r in missing if r["missing"] == 1],
              "two_missing": [r for r in missing if r["missing"] == 2],
              "incomplete_anchor_1_gold": [r for r in missing if r["anchor_gold"][0]],
              "incomplete_anchor_1_wrong": [r for r in missing if not r["anchor_gold"][0]],
              "incomplete_anchor_1_wrong_anchor_2_gold": [r for r in missing
                                                         if not r["anchor_gold"][0] and r["anchor_gold"][1]]}
    return {**meta, "gpus": dict(sorted(Counter(s["gpu"] for s in shards).items())),
            "cohorts": {k: panel(v) for k, v in groups.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    for key in ("fixture", "pools", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--write-replay", type=Path)
    a = p.parse_args()
    shards = [unpack(p) for p in sorted(a.input.glob("coverage-*.json.gz"))] if a.input.is_dir() else unpack(a.input)
    report = summarize(shards, a.fixture, a.pools)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        data = json.dumps(shards, separators=(",", ":"), allow_nan=False).encode()
        a.write_replay.write_bytes(gzip.compress(data, mtime=0))


if __name__ == "__main__":
    main()
