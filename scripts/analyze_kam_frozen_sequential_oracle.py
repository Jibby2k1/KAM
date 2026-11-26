#!/usr/bin/env python3
"""Replay support and joint-exact oracle gains on fixed Wikipedia pools."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_sentence_verifier import paired

ARMS = ("broad", "sequential")
METRICS = ("support", "answer_exact", "joint_exact")


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def score(rows, sealed, holdout):
    outputs = {a: {m: [] for m in METRICS} for a in ARMS}
    for r in rows:
        for arm, metrics in outputs.items():
            state = r["arms"][arm]
            metrics["support"].append(state["support"])
            metrics["answer_exact"].append(state["exact"])
            metrics["joint_exact"].append(state["support"] and state["exact"])
    broad, sequential = (outputs[a] for a in ARMS)
    oracle = {m: [a or b for a, b in zip(broad[m], sequential[m], strict=True)] for m in METRICS}
    return {
        "n": len(rows),
        "sequential_first_is_gold": sum(
            sealed[r["id"]][r["arms"]["sequential"]["pair"][0]] in holdout[r["id"]] for r in rows
        ),
        "same_pair": sum(set(r["arms"]["broad"]["pair"]) == set(r["arms"]["sequential"]["pair"]) for r in rows),
        "same_reader_order": sum(r["arms"]["broad"]["pair"] == r["arms"]["sequential"]["pair"] for r in rows),
        "correct": {a: {m: sum(values) for m, values in o.items()} for a, o in (outputs | {"oracle": oracle}).items()},
        "exclusive_successes": {
            m: {"broad_only": sum(a and not b for a, b in zip(broad[m], sequential[m], strict=True)),
                "sequential_only": sum(b and not a for a, b in zip(broad[m], sequential[m], strict=True))}
            for m in METRICS
        },
        "sequential_vs_broad": {m: paired(sequential[m], broad[m]) for m in METRICS},
        "oracle_vs_broad": {m: paired(oracle[m], broad[m]) for m in METRICS},
        "oracle_vs_sequential": {m: paired(oracle[m], sequential[m]) for m in METRICS},
    }


def summarize(shards, fixture_path, pool_path, broad_path):
    fields = ("fixture_sha256", "pools_sha256", "broad_replay_sha256", "baseline_sha256", "index_sha256", "shards")
    meta = shards[0]
    assert all(all(s[k] == meta[k] for k in fields) for s in shards)
    assert sorted(s["shard"] for s in shards) == list(range(meta["shards"]))
    fixture = gzip.decompress(fixture_path.read_bytes())
    assert meta["fixture_sha256"] == sha256(fixture).hexdigest()
    assert meta["pools_sha256"] == digest(pool_path)
    assert meta["broad_replay_sha256"] == digest(broad_path)
    sealed = unpack(pool_path)
    assert meta["fixture_sha256"] == sealed["fixture_sha256"] and meta["index_sha256"] == sealed["index_sha256"]
    prior = {r["id"]: r for shard in unpack(broad_path) for r in shard["rows"] if r["split"] == "test"}
    holdout = {r["id"]: set(r["gold_ids"]) for r in json.loads(fixture)["rows"]["test"]}
    rows = sorted([r for s in shards for r in s["rows"]], key=lambda r: r["id"])
    assert len(rows) == len(holdout) == len(prior) == 1024
    assert {r["id"] for r in rows} == holdout.keys() == prior.keys()
    reread_differences = []
    for row in rows:
        id_ = row["id"]
        ordered = sealed["rows"][id_]
        assert row["pool_hash"] == prior[id_]["pool_hash"]
        assert row["pool_hash"] == sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()
        assert set(row["arms"]) == set(ARMS)
        for arm in row["arms"].values():
            p = arm["pair"]
            assert len(p) == len(set(p)) == 2 and all(0 <= i < 64 for i in p)
            assert arm["support"] == ({ordered[i] for i in p} == holdout[id_])
        b, s = (row["arms"][a] for a in ARMS)
        original = prior[id_]["arms"]["broad"]
        assert b["pair"] == original["pair"] and b["support"] == original["support"]
        assert b["pair"] != s["pair"] or b == s
        if b["exact"] != original["exact"]:
            reread_differences.append(id_)
    panels = {
        "all": rows,
        "gold_in_pool": [r for r in rows if prior[r["id"]]["gold_in_pool"] == 2],
        "natural_linked_distractor": [r for r in rows if prior[r["id"]]["gold_linked"] is True
                                      and prior[r["id"]]["pool_link_count"] > 1],
        "gold_unlinked": [r for r in rows if prior[r["id"]]["gold_linked"] is False],
        "missing_gold": [r for r in rows if prior[r["id"]]["gold_in_pool"] < 2],
        "pair_disagreement": [r for r in rows if set(r["arms"]["broad"]["pair"]) != set(r["arms"]["sequential"]["pair"])],
    }
    return {**{k: meta[k] for k in fields}, "gpu": sorted({s["gpu"] for s in shards}),
            "broad_reader_differences_from_previous_pass": reread_differences,
            "tests": {name: score(items, sealed["rows"], holdout) for name, items in panels.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    for name in ("fixture", "pools", "broad-replay", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--write-replay", type=Path)
    a = p.parse_args()
    shards = [unpack(p) for p in sorted(a.input.glob("eval-*.json.gz"))] if a.input.is_dir() else unpack(a.input)
    report = summarize(shards, a.fixture, a.pools, a.broad_replay)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        data = json.dumps(shards, separators=(",", ":"), allow_nan=False).encode()
        a.write_replay.write_bytes(gzip.compress(data, mtime=0))


if __name__ == "__main__":
    main()
