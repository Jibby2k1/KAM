#!/usr/bin/env python3
"""Replay the dense and hybrid initial pools through fixed broad and expanded policies."""

import argparse
from hashlib import sha256
import gzip
import json
from pathlib import Path

from analyze_kam_sentence_verifier import paired

METHODS = ("dense", "hybrid")
POLICIES = ("broad", "affinity_anchor", "broad_anchor")
MEASURES = ("coverage", "shortlist", "support", "joint_em")


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def audit_method(study, method, fixture, fixture_hash):
    path = study / f"pools-{method}.json.gz"
    frozen = unpack(path)
    assert frozen["fixture_sha256"] == fixture_hash and frozen["method"] == method
    assert frozen["pool_depth"] == 64 and frozen["component_depth"] == 128
    assert frozen["rows"].keys() == fixture.keys()
    base = study / f"{method}-affinity-replay.json.gz"
    refs = unpack(base)
    alternatives = unpack(study / f"{method}-broad-replay.json.gz")
    keys = ("fixture_sha256", "pools_sha256", "config_sha256", "index_sha256", "fitted_sha256", "heads_sha256",
            "ablations_sha256", "baseline_sha256", "models", "indexed_documents", "guard_columns", "split")
    provenance = {k: refs[0][k] for k in keys}
    assert all(all(s[k] == v for k, v in provenance.items()) for s in (*refs, *alternatives))
    assert provenance["pools_sha256"] == digest(path)
    assert provenance["fixture_sha256"] == fixture_hash
    assert all(provenance[k] == frozen[k] for k in ("index_sha256", "config_sha256"))
    assert all(s["anchor_reference_sha256"] == digest(base) for s in alternatives)
    for shards in (refs, alternatives):
        assert len(shards) == 4 and sorted(s["shard"] for s in shards) == list(range(4))
        assert sum(len(s["rows"]) for s in shards) == len(fixture)
    initial, ranked = ({r["id"]: r for s in group for r in s["rows"]} for group in (refs, alternatives))
    assert initial.keys() == ranked.keys() == fixture.keys()
    matches, result = 0, {}
    for id_, gold in fixture.items():
        pool = frozen["rows"][id_]
        original = set(pool)
        assert len(original) == len(pool) == 64 and len(gold) == 2
        first, alt = initial[id_], ranked[id_]
        assert first["pool_hash"] == alt["pool_hash"] == sha256(
            json.dumps(pool, separators=(",", ":")).encode()).hexdigest()
        assert alt["first"] == first["arms"]["broad"]["pair"][0] and first["first"] in original
        matches += first["arms"]["broad"] == alt["arms"]["broad"]
        result[id_] = {"missing": len(gold - original)}
        for policy, source, arm in (("broad", alt, "broad"),
                                    ("affinity_anchor", first, "conditioned_union"),
                                    ("broad_anchor", alt, "conditioned_union")):
            added = source["corpus_hits"]["conditioned_union"] if policy != "broad" else []
            if policy != "broad":
                assert len(added) == len(set(added)) == 64 and source["first"] not in added
            search = original | set(added)
            items = source["arms"][arm]
            pair = set(items["pair"])
            assert len(pair) == 2 and pair <= search
            support = pair == gold
            assert support <= items["gold_pair_in_shortlist"] <= (gold <= search)
            result[id_][policy] = {"coverage": gold <= search, "shortlist": items["gold_pair_in_shortlist"],
                                   "support": support, "joint_em": support and items["exact"]}
    return result, provenance | {"repeated_broad_matches": matches, "rank_constant": frozen["rrf_rank_constant"],
                                  "affinity_gpus": [s["gpu"] for s in refs],
                                  "broad_anchor_gpus": [s["gpu"] for s in alternatives]}


def summarize(rows):
    n = len(rows)
    all_labels = {policy: {m: [r[method][p][m] for r in rows] for m in MEASURES}
                  for method in METHODS for p in POLICIES for policy in (method + "_" + p,)}
    pairs = [("hybrid_" + p, "dense_" + p) for p in POLICIES]
    pairs += [(m + "_broad_anchor", m + "_broad") for m in METHODS]
    pairs += [(m + "_broad_anchor", m + "_affinity_anchor") for m in METHODS]
    return {"questions": n,
            "totals": {p: {m: sum(vals) for m, vals in measures.items()} for p, measures in all_labels.items()},
            "paired": {h + "_vs_" + d: {m: paired(all_labels[h][m], all_labels[d][m]) for m in MEASURES}
                       for h, d in pairs} if n else {}}


def evaluate(study):
    blob = gzip.decompress((study / "fixture.json.gz").read_bytes())
    items = json.loads(blob)["rows"]["test"]
    fixture = {q["id"]: set(q["gold_ids"]) for q in items}
    assert len(items) == len(fixture) == 1024
    labels, provenance = {}, {}
    for method in METHODS:
        labels[method], provenance[method] = audit_method(study, method, fixture, sha256(blob).hexdigest())
    fixed = ("index_sha256", "config_sha256", "fitted_sha256", "heads_sha256", "ablations_sha256",
             "baseline_sha256", "models", "indexed_documents", "rank_constant", "split")
    assert all(provenance["dense"][key] == provenance["hybrid"][key] for key in fixed)
    assert all(p["broad_anchor_gpus"] == ["NVIDIA B200"] * 4 for p in provenance.values())
    rows = [{method: labels[method][i] for method in METHODS} for i in sorted(fixture)]
    cross = [[sum(r["dense"]["missing"] == d and r["hybrid"]["missing"] == h for r in rows) for h in range(3)]
             for d in range(3)]
    groups = {"all": rows}
    for method in METHODS:
        groups[method + "_incomplete"] = [r for r in rows if r[method]["missing"] > 0]
        for k, label in enumerate(("complete", "one_missing", "two_missing")):
            groups[method + "_" + label] = [r for r in rows if r[method]["missing"] == k]
    return {"provenance": provenance, "initial_missing_count_table_dense_by_hybrid": cross,
            "cohorts": {g: summarize(cases) for g, cases in groups.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(evaluate(a.study), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
