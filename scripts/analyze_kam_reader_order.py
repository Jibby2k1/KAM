#!/usr/bin/env python3
"""Measure the paired EM effect of reversing the fixed reader input order."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_sentence_verifier import paired

POLICIES = ("reference", "reversed")


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def analyze(shards, fixture_path, reference_path):
    fixture = gzip.decompress(fixture_path.read_bytes())
    refs = read(reference_path)
    provenance = {k: refs[0][k] for k in ("fixture_sha256", "index_sha256", "models")}
    assert provenance["fixture_sha256"] == sha256(fixture).hexdigest()
    assert all(all(shard[k] == v for k, v in provenance.items()) for shard in (*refs, *shards))
    digest = sha256(reference_path.read_bytes()).hexdigest()
    assert all(s["reference_sha256"] == digest for s in shards)
    assert sorted(s["shard"] for s in shards) == list(range(4))
    gold = {q["id"]: set(q["gold_ids"]) for q in json.loads(fixture)["rows"]["test"]}
    reference = {r["id"]: r["arms"]["conditioned_union"] for s in refs for r in s["rows"]}
    rows = sorted([r for s in shards for r in s["rows"]], key=lambda r: r["id"])
    assert len(rows) == len(gold) == 1024 and gold.keys() == reference.keys() == {r["id"] for r in rows}
    parity = 0
    enriched = []
    for r in rows:
        key = r["id"]
        assert r["pair"] == reference[key]["pair"] and len(set(r["pair"])) == 2
        assert set(r["exact"]) == set(POLICIES) and all(type(v) is bool for v in r["exact"].values())
        parity += r["exact"]["reference"] == reference[key]["exact"]
        enriched.append(r | {"support": set(r["pair"]) == gold[key]})

    def summarize(cases):
        support = sum(r["support"] for r in cases)
        measures = {}
        for metric in ("exact", "joint_em"):
            old, new = ([r["exact"][arm] and (metric == "exact" or r["support"]) for r in cases] for arm in POLICIES)
            measures[metric] = {"reference": sum(old), "reversed": sum(new),
                                "either_order_oracle": sum(a or b for a, b in zip(old, new, strict=True)),
                                **paired(new, old)}
        return {"questions": len(cases), "support_pairs": support, **measures}

    subsets = {"all": enriched, "both_supports": [r for r in enriched if r["support"]],
               "missing_support": [r for r in enriched if not r["support"]]}
    return {**provenance, "reference_sha256": digest, "archived_em_parity": parity,
            "gpus": dict(Counter(s["gpu"] for s in shards)),
            "cohorts": {k: summarize(v) for k, v in subsets.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    for name in ("fixture", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--write-replay", type=Path)
    a = p.parse_args()
    shards = ([read(x) for x in sorted(a.input.glob("order-*.json.gz"))] if a.input.is_dir() else read(a.input))
    result = analyze(shards, a.fixture, a.reference)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        a.write_replay.write_bytes(gzip.compress(json.dumps(shards, separators=(",", ":"), allow_nan=False).encode(),
                                                 mtime=0))


if __name__ == "__main__":
    main()
