#!/usr/bin/env python3
"""Audit final evidence selection after fitting on separate MuSiQue questions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_musique_final_reader import METHODS
from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

MODES = ("hybrid_rank", *(f"learned_{name}/{i}" for i in range(3) for name in ("path", "selector", "oracle")))
KEYS = ("support", "em", "joint")


def summarize(cases, records):
    scores = {name: {key: [] for key in KEYS} for name in MODES}
    for q in cases:
        gold = set(q["gold"])
        rows = records[q["id"]]["arms"]
        for mode, a in scores.items():
            selected = rows[mode]
            pred = selected["evidence_ids"]
            assert len(pred) == len(set(pred)) == 4
            covered = gold <= set(pred)
            em = bool(selected["em"])
            for key, flag in zip(KEYS, (covered, em, covered and em), strict=True):
                a[key].append(flag)
    n = shared_evidence_clusters(cases)
    paired = {}
    for seed in range(3):
        primary = f"learned_selector/{seed}"
        compared = (f"learned_path/{seed}", "hybrid_rank")
        paired[str(seed)] = {arm: {k: compare(scores[primary][k], scores[arm][k], n) for k in KEYS} for arm in compared}
    return {"questions": len(cases),
            "totals": {k: {m: sum(seq) for m, seq in metrics.items()} for k, metrics in scores.items()},
            "paired": paired}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "policy", "picks", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in ("old", "selector"):
        p.add_argument("--" + name, required=True, nargs=4, type=Path)
    a = p.parse_args()
    digest = lambda p: sha256(p.read_bytes()).hexdigest()
    fixture = unpack(a.fixture)
    policy = json.loads(a.policy.read_text())
    picks = json.loads(a.picks.read_text())
    assert picks["policy_sha256"] == digest(a.policy)
    assert policy["test_fixture_sha256"] == picks["fixture_sha256"] == fixture["fixture_sha256"]
    arms = [sorted(map(unpack, paths), key=lambda c: c["shard"]) for paths in (a.old, a.selector)]
    assert all([q["shard"] for q in rows] == list(range(4)) for rows in arms)
    choices = {q["id"]: q["selections"] for q in picks["rows"]}
    exact_parity = Counter()
    metadata = ("fixture_sha256", "corpus_sha256", "baseline_sha256", "learned_sha256", "label_sha256", "reader_revision")
    output = {}
    for past, new in zip(*arms, strict=True):
        assert (past["shard"], past["shards"]) == (new["shard"], new["shards"])
        assert new["fixture_sha256"] == fixture["fixture_sha256"] and new["device"] == "NVIDIA L4"
        assert all(past[k] == new[k] for k in metadata)
        assert new["selector_picks_sha256"] == digest(a.picks) and new["selector_policy_sha256"] == digest(a.policy)
        for original, current, case in zip(past["rows"], new["rows"], fixture["rows"][new["shard"]::4], strict=True):
            id_ = case["id"]
            assert original["id"] == current["id"] == id_
            before, after = original["arms"], current["arms"]
            for mode in METHODS:
                assert before[mode]["evidence_ids"] == after[mode]["evidence_ids"]
                exact_parity[mode] += before[mode] == after[mode]
            for i, ids in enumerate(choices[id_]):
                assert after[f"learned_selector/{i}"]["evidence_ids"] == ids
            output[id_] = current
    cases = fixture["rows"]
    scope = {"all": cases, **{str(i): [q for q in cases if q["hops"] == i] for i in COHORTS}}
    out = {"fixture_sha256": fixture["fixture_sha256"], "old_measured_exact_parity": exact_parity,
           "queries_replayed": len(output), "scopes": {name: summarize(group, output) for name, group in scope.items()}}
    if "models" in policy:
        out["fit_questions_used"] = [model["fit_questions"] for model in policy["models"]]
    else:
        out["fixed_policy"] = policy["method"]
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
