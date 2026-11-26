#!/usr/bin/env python3
"""Audit three independent affinity-anchor expansion cohorts and summarize paired effects."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

STUDIES = {
    "conditioned_union_v1": "replay.json.gz",
    "expansion_guard_v1": "replay.json.gz",
    "broad_anchor_v1": "affinity-replay.json.gz",
}
ARMS = ("broad", "conditioned_union")
METRICS = ("coverage", "shortlist", "support", "joint_em")
FROZEN = ("index_sha256", "config_sha256", "fitted_sha256", "heads_sha256", "ablations_sha256",
          "baseline_sha256", "models", "indexed_documents")


def unpack(p):
    return json.loads(gzip.decompress(p.read_bytes()))


def digest(b):
    return sha256(b).hexdigest()


def read(study, replay_name):
    paths = {k: study / n for k, n in (("fixture", "fixture.json.gz"), ("pools", "pools.json.gz"),
                                     ("replay", replay_name))}
    raw = gzip.decompress(paths["fixture"].read_bytes())
    cases = json.loads(raw)["rows"]["test"]
    pool_data = unpack(paths["pools"])
    shards = unpack(paths["replay"])
    assert len(shards) == 4 and sorted(s["shard"] for s in shards) == list(range(4))
    meta = {k: shards[0][k] for k in FROZEN}
    assert all(all(s[k] == v for k, v in meta.items()) for s in shards)
    assert all(s["fixture_sha256"] == pool_data["fixture_sha256"] == digest(raw) and
               s["pools_sha256"] == digest(paths["pools"].read_bytes()) and
               s["index_sha256"] == pool_data["index_sha256"] for s in shards)
    pools = pool_data["rows"]
    records = {r["id"]: r for s in shards for r in s["rows"]}
    ids = {q["id"] for q in cases}
    titles = {t for q in cases for t in q["gold_titles"]}
    texts = {" ".join(re.findall(r"\w+", q["question"].casefold())) for q in cases}
    assert len(ids) == len(texts) == len(cases) == 1024 and len(titles) == 2048
    assert ids == pools.keys() == records.keys()
    assert sum(len(s["rows"]) for s in shards) == 1024
    out = []
    for q in cases:
        r = records[q["id"]]
        pool = pools[q["id"]]
        gold = set(q["gold_ids"])
        assert None not in gold and len(gold) == 2
        assert len(pool) == len(set(pool)) == 64
        assert r["pool_hash"] == digest(json.dumps(pool, separators=(",", ":")).encode())
        first = r["first"]
        hits = r["corpus_hits"]["conditioned_union"]
        assert first in pool and first not in hits and len(hits) == len(set(hits)) == 64
        row = {"missing": len(gold - set(pool)), "anchor_gold": first in gold}
        for arm, candidate in (("broad", set(pool)), ("conditioned_union", set(pool) | set(hits))):
            a = r["arms"][arm]
            pair = set(a["pair"])
            assert len(pair) == 2 and pair <= candidate
            c, s, support = gold <= candidate, bool(a["gold_pair_in_shortlist"]), gold == pair
            assert support <= s <= c
            row[arm] = {"coverage": c, "shortlist": s, "support": support, "joint_em": support and a["exact"]}
        out.append(row)
    return (ids, titles, texts), out, {"frozen": meta,
                                      **{k + "_sha256": digest(p.read_bytes()) for k, p in paths.items()}}


def estimate(metrics):
    """Pair rows within each fixed cohort and resample within cohort only."""
    n = sum(len(pairs) for pairs in metrics)
    if not n:
        return None
    boot = np.zeros(5000, dtype=np.int64)
    rng = np.random.default_rng(55031)
    wins = losses = old = new = 0
    for cohort in metrics:
        if not cohort:
            continue
        a, b = np.asarray(cohort, dtype=np.int64).T
        delta = b - a
        counts = np.bincount(delta + 1, minlength=3)
        sample = rng.multinomial(len(cohort), counts / len(cohort), size=len(boot))
        boot += sample[:, 2] - sample[:, 0]
        losses += int(counts[0])
        wins += int(counts[2])
        old += int(a.sum())
        new += int(b.sum())
    lo, hi = np.quantile(boot / n, [.025, .975])
    return {"broad": old, "conditioned_union": new, "wins": wins, "losses": losses,
            "delta": round((new - old) / n, 8), "bootstrap95": [round(float(lo), 8), round(float(hi), 8)]}


def summarize(groups):
    return {"n": sum(len(g) for g in groups),
            "metrics": {m: estimate([[(r[ARMS[0]][m], r[ARMS[1]][m]) for r in g] for g in groups]) for m in METRICS}}


def evaluate(base):
    all_rows, provenance = {}, {}
    seen = [set(), set(), set()]
    for name, replay in STUDIES.items():
        keys, all_rows[name], provenance[name] = read(base / name, replay)
        for prev, current in zip(seen, keys, strict=True):
            assert prev.isdisjoint(current), (name, len(prev & current))
            prev.update(current)
    frozen = list(provenance.values())
    assert all(p["frozen"] == frozen[0]["frozen"] for p in frozen)
    conditions = {"all": lambda r: True, "complete": lambda r: r["missing"] == 0,
                  "incomplete": lambda r: r["missing"] > 0, "one_missing": lambda r: r["missing"] == 1,
                  "two_missing": lambda r: r["missing"] == 2,
                  "incomplete_anchor_gold": lambda r: r["missing"] > 0 and r["anchor_gold"],
                  "incomplete_anchor_wrong": lambda r: r["missing"] > 0 and not r["anchor_gold"]}
    return {"provenance": provenance, "unique_query_ids": len(seen[0]), "unique_gold_titles": len(seen[1]),
            "unique_normalized_questions": len(seen[2]),
            "per_cohort": {name: {label: summarize([[r for r in rows if cond(r)]])
                                  for label, cond in conditions.items()} for name, rows in all_rows.items()},
            "combined": {label: summarize([[r for r in rows if cond(r)] for rows in all_rows.values()])
                         for label, cond in conditions.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(evaluate(a.base), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
