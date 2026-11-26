#!/usr/bin/env python3
"""Audit paired sentence-selection outcomes without changing retrieval."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_sentence_verifier import paired

ARMS = ("prefix", "lead_two", "dense_two", "bridge_two")


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def summarize(rows):
    totals = {a: {"exact": sum(r["exact"][a] for r in rows),
                  "joint_em": sum(r["exact"][a] and r["support"] for r in rows),
                  "visible_annotations": sum(r["visible"][a] for r in rows),
                  "questions_with_all_annotations_visible": sum(
                      r["visible"][a] == r["valid_annotations"] for r in rows)} for a in ARMS}
    contrasts = {}
    for a, b in (("lead_two", "prefix"), ("dense_two", "prefix"), ("bridge_two", "prefix"),
                 ("dense_two", "lead_two"), ("bridge_two", "dense_two")):
        if rows:
            contrasts[f"{a}_vs_{b}"] = {
                metric: paired([r["exact"][a] and (metric == "exact" or r["support"]) for r in rows],
                               [r["exact"][b] and (metric == "exact" or r["support"]) for r in rows])
                for metric in ("exact", "joint_em")
            }
    oracle = sum(r["support"] and any(r["exact"][a] for a in ("prefix", "dense_two", "bridge_two")) for r in rows)
    return {"questions": len(rows), "support_pairs": sum(r["support"] for r in rows), "joint_oracle": oracle,
            "valid_annotations": sum(r["valid_annotations"] for r in rows), "totals": totals, "contrasts": contrasts}


def evaluate(shards, fixture, evidence, reference):
    fixture_data, evidence_data = (gzip.decompress(p.read_bytes()) for p in (fixture, evidence))
    fx, ev = json.loads(fixture_data), json.loads(evidence_data)
    digest = lambda b: sha256(b).hexdigest()
    fixture_sha = digest(fixture_data)
    assert ev["fixture_sha256"] == fixture_sha and ev["dataset_sha256"] == fx["external_test_sha256"]
    refs = read(reference)
    assert len(shards) == 4 and sorted(s["shard"] for s in shards) == list(range(4))
    keys = ("fixture_sha256", "index_sha256", "models")
    provenance = {k: refs[0][k] for k in keys}
    assert provenance["fixture_sha256"] == fixture_sha
    assert all(all(s[k] == v for k, v in provenance.items()) for s in (*shards, *refs))
    source_hashes = {"evidence_sha256": digest(evidence_data), "reference_sha256": digest(reference.read_bytes())}
    assert all(all(s[k] == v for k, v in source_hashes.items()) for s in shards)

    candidates = {q["id"]: (set(q["gold_ids"]), q) for q in fx["rows"]["test"]}
    annotated = {r["id"]: r["articles"] for r in ev["rows"]}
    retrieved = {r["id"]: r["arms"]["conditioned_union"] for s in refs for r in s["rows"]}
    rows = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == len(candidates) == 1024
    assert {r["id"] for r in rows} == candidates.keys() == annotated.keys() == retrieved.keys()
    audited = []
    prefix_parity = 0
    for r in rows:
        id_ = r["id"]
        gold, _ = candidates[id_]
        chosen = retrieved[id_]
        assert len(gold) == 2 and r["pair"] == chosen["pair"] and len(set(r["pair"])) == 2
        assert set(r["visible"]) == set(r["exact"]) == set(ARMS) and all(type(b) is bool for b in r["exact"].values())
        evidence_rows = annotated[id_]
        assert {doc["row_id"] for doc in evidence_rows} == gold
        assert r["valid_annotations"] == sum(len(doc["support_positions"]) for doc in evidence_rows)
        retrieved_limit = sum(len(doc["support_positions"]) for doc in evidence_rows if doc["row_id"] in r["pair"])
        assert all(0 <= n <= retrieved_limit for n in r["visible"].values())
        prefix_parity += r["exact"]["prefix"] == chosen["exact"]
        audited.append({**r, "support": set(r["pair"]) == gold})
    groups = {"all": audited, "both_supports_retrieved": [r for r in audited if r["support"]],
              "some_support_missing": [r for r in audited if not r["support"]]}
    return {**provenance, **source_hashes, "gpu_shards": dict(Counter(s["gpu"] for s in shards)),
            "prefix_parity_with_reference": prefix_parity,
            "cohorts": {k: summarize(v) for k, v in groups.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    for name in ("fixture", "evidence", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--write-replay", type=Path)
    a = p.parse_args()
    shards = ([read(x) for x in sorted(a.input.glob("selection-*.json.gz"))] if a.input.is_dir() else read(a.input))
    result = evaluate(shards, a.fixture, a.evidence, a.reference)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        a.write_replay.write_bytes(gzip.compress(json.dumps(shards, separators=(",", ":"), allow_nan=False).encode(),
                                                 mtime=0))


if __name__ == "__main__":
    main()
