#!/usr/bin/env python3
"""Fit the frozen shortlist features on title-disjoint native 2Wiki questions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analyze_kam_sentence_verifier import fit, select
from analyze_kam_shared_serving import unpack
from audit_kam_natural_data import digest, words

ARM = "conditioned128"
PENALTY = 0.025


def load_features(paths, fixture_path):
    raw = fixture_path.read_bytes()
    fixture = json.loads(raw)
    chunks = sorted(map(unpack, paths), key=lambda c: c["shard"])
    columns = chunks[0]["columns"]
    assert len(columns) == len(set(columns)) == 16
    assert [c["shard"] for c in chunks] == list(range(4))
    frozen = {k: chunks[0][k] for k in ("columns", "models", "search_hashes", "fitted_sha256", "corpus")}
    assert all(c["fixture_sha256"] == sha256(raw).hexdigest() and c["source_sha256"] == fixture["source_sha256"]
               and c["shards"] == 4 and c["device"] == "NVIDIA L4"
               and all(c[k] == v for k, v in frozen.items()) for c in chunks)
    for c in chunks:
        assert [r["id"] for r in c["rows"]] == [r["id"] for r in fixture["rows"][c["shard"]::4]]
    indices = {row["id"]: row["arms"][ARM] for c in chunks for row in c["rows"]}
    for row in fixture["rows"]:
        gold = {u["corpus_id"] for u in row["units"]}
        variant = indices[row["id"]]
        assert len(variant["pairs"]) == len(variant["features"]) > 1
        assert all(len(v) == len(columns) for v in variant["features"])
        assert variant["gold"] == next((i for i, p in enumerate(variant["pairs"]) if set(p) == gold), -1)
    return fixture, indices, frozen


def separation(train, test):
    train_ids = {r["id"] for r in train}
    test_ids = {r["id"] for r in test}
    train_titles = {u["title_sha256"] for r in train for u in r["units"]}
    test_titles = {u["title_sha256"] for r in test for u in r["units"]}
    assert len(train_ids) == len(train) == 512 and len(test_ids) == len(test) == 1024
    assert len(train_titles) == 1024 and len(test_titles) == 2048
    assert train_ids.isdisjoint(test_ids) and train_titles.isdisjoint(test_titles)
    assert {tuple(words(r["question"])) for r in train}.isdisjoint(tuple(words(r["question"])) for r in test)
    return {"fit_questions": len(train_ids), "test_questions": len(test_ids),
            "fit_supporting_titles": len(train_titles), "test_supporting_titles": len(test_titles)}


def fit_policy(a):
    fixture, records, provenance = load_features(a.shards, a.fixture)
    holdout = json.loads(a.holdout.read_text())
    assert fixture["held_out_fixture_sha256"] == digest(a.holdout)
    audit = separation(fixture["rows"], holdout["rows"])
    center, scale, coefficients, eligible = fit(
        [{"variants": {ARM: records[r["id"]]}} for r in fixture["rows"]], 16, (ARM,), penalty=PENALTY)
    return {"policy": {"mean": center.tolist(), "scale": scale.tolist(), "coefficients": coefficients.tolist(),
                       "columns": provenance["columns"], "penalty": PENALTY},
            "audit": audit | {"included_shortlisted_positive": eligible},
            "fit_fixture_sha256": digest(a.fixture), "test_fixture_sha256": digest(a.holdout),
            "fit_features_sha256": sorted(map(digest, a.shards)), **provenance}


def choices(a):
    fixture, records, provenance = load_features(a.shards, a.fixture)
    with a.lock.open() as file:
        model = json.load(file)
    assert model["test_fixture_sha256"] == digest(a.fixture)
    assert all(provenance[k] == model[k] for k in ("columns", "fitted_sha256", "corpus", "models"))
    assert provenance["search_hashes"] != model["search_hashes"]
    assert digest(a.reference) == provenance["fitted_sha256"]
    original = json.loads(a.reference.read_text())["native_semantic"]

    def params(p):
        return (np.asarray(p["mean"]), np.asarray(p["scale"]), np.asarray(p["coefficients"]), 0)

    fitted = {"frozen": params(original), "refit": params(model["policy"])}
    rows = []
    for r in fixture["rows"]:
        x = records[r["id"]]
        picks = {mode: x["pairs"][select(x, p)] for mode, p in fitted.items()}
        rows.append({"id": r["id"], "pairs": picks, "shortlisted": x["gold"] >= 0})
    return {"fixture_sha256": digest(a.fixture), "policy_sha256": digest(a.lock),
            "feature_sha256": sorted(map(digest, a.shards)), **provenance, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "score"))
    p.add_argument("--shards", nargs=4, required=True, type=Path)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--holdout", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--lock", type=Path)
    a = p.parse_args()
    if a.mode == "fit":
        assert a.holdout is not None
        result = fit_policy(a)
    else:
        assert a.reference is not None and a.lock is not None
        result = choices(a)
    a.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
