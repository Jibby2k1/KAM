#!/usr/bin/env python3
"""Audit original-question exact match at a shared four-paragraph budget."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

SEEDS = range(3)
CONTROLS = ("hybrid_rank", "hybrid_dense", "hybrid_oracle", "gold_only", "no_evidence")
METHODS = (*CONTROLS, *(f"learned_{m}/{seed}" for m in ("path", "dense", "oracle") for seed in SEEDS))


def assess(cases, records):
    totals = {arm: Counter() for arm in METHODS}
    flags = {arm: {k: [] for k in ("all_support", "answer_em", "joint")} for arm in METHODS}
    for q in cases:
        arms = records[q["id"]]["arms"]
        assert set(arms) == set(METHODS)
        gold = set(q["gold"])
        for arm, a in arms.items():
            ids = a["evidence_ids"]
            assert len(ids) == len(set(ids))
            assert len(ids) == (0 if arm == "no_evidence" else len(gold) if arm == "gold_only" else 4)
            covered = gold <= set(ids)
            em = bool(a["em"])
            values = {"all_support": covered, "answer_em": em, "joint": covered and em}
            totals[arm].update({"support_count": len(gold.intersection(ids)), **values})
            for k, v in values.items():
                flags[arm][k].append(v)
    groups = shared_evidence_clusters(cases)
    pairs = [(f"learned_{name}/{i}", other) for i in SEEDS
             for name, other in (("path", "hybrid_rank"), ("dense", "hybrid_dense"), ("oracle", "hybrid_oracle"))]
    comparisons = {f"{a}_vs_{b}": {key: compare(flags[a][key], flags[b][key], groups)
                                     for key in flags[a]} for a, b in pairs}
    return {"questions": len(cases), "metrics": totals, "paired": comparisons}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("reader", "baseline"):
        p.add_argument("--" + name, type=Path, nargs=4, required=True)
    p.add_argument("--learned", action="append", nargs=4, type=Path, required=True)
    a = p.parse_args()
    fixture = unpack(a.manifest)
    by_shard = lambda s: sorted(s, key=lambda part: part["shard"])
    read = by_shard(map(unpack, a.reader))
    baseline = {sha256(file.read_bytes()).hexdigest(): unpack(file) for file in a.baseline}
    assert len(a.learned) == 3
    learned = [{sha256(file.read_bytes()).hexdigest(): unpack(file) for file in files} for files in a.learned]
    assert [c["shard"] for c in read] == list(range(4))
    assert len({c["label_sha256"] for c in read}) == len({c["reader_revision"] for c in read}) == 1
    results = {}
    for part in read:
        index = part["shard"]
        original = baseline[part["baseline_sha256"]]
        ranks = [inputs[hash_] for inputs, hash_ in zip(learned, part["learned_sha256"], strict=True)]
        assert all((item["shard"], item["shards"]) == (index, 4) and
                   item["fixture_sha256"] == fixture["fixture_sha256"] and item["corpus_sha256"] == fixture["corpus_sha256"]
                   for item in (part, original, *ranks))
        assert part["device"] == "NVIDIA L4"
        for q, prior, row, *fit in zip(fixture["rows"][index::4], original["rows"], part["rows"],
                                      *(r["rows"] for r in ranks), strict=True):
            id_ = q["id"]
            assert id_ == row["id"] == prior["id"] and all(k["id"] == id_ for k in fit)
            arms = row["arms"]
            assert arms["hybrid_rank"]["evidence_ids"] == prior["hybrid"][:4]
            assert set(arms["gold_only"]["evidence_ids"]) == set(q["gold"])
            for seed, selected in enumerate(fit):
                for name in ("path", "dense", "oracle"):
                    choice = arms[f"learned_{name}/{seed}"]["evidence_ids"]
                    assert set(choice) <= set(selected["pool"])
                if selected["fallback"]:
                    assert arms[f"learned_path/{seed}"] == arms["hybrid_rank"]
                oracle = arms[f"learned_oracle/{seed}"]["evidence_ids"]
                assert set(q["gold"]).intersection(oracle) == set(q["gold"]).intersection(selected["pool"])
            results[id_] = row
    cases = fixture["rows"]
    scopes = {"all": cases, **{str(i): [q for q in cases if q["hops"] == i] for i in COHORTS}}
    out = {"fixture_sha256": fixture["fixture_sha256"], "reader_prompts": sum(c["unique_prompts"] for c in read),
           "scopes": {k: assess(panel, results) for k, panel in scopes.items()}}
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
