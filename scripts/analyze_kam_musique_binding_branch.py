#!/usr/bin/env python3
"""Audit final-step MuSiQue alternatives and fixed-evidence exactness."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

ARMS = ("top1", "branch_unbound", "branch_span")
BUDGETS = {2: 32, 3: 21, 4: 16}
EXTRA = {"fixed_evidence_ids", "fixed_answer_em", "alternative_search", "alternate_final_rank"}
METRICS = ("complete_pool", "fixed_support", "fixed_answer", "fixed_joint", "active")


def decision(arm, truth):
    matching = truth <= set(arm["fixed_evidence_ids"])
    exact = bool(arm["fixed_answer_em"])
    return (truth <= set(arm["pool"]), matching, exact, matching and exact, arm["alternative_search"])


def assess(scopes, rows):
    output = {}
    for key, cases in scopes.items():
        groups = shared_evidence_clusters(cases)
        output[str(key)] = {"questions": len(cases), "clusters": len(groups), "seeds": {}}
        for seed, records in rows.items():
            outcomes = {m: [] for m in ARMS}
            for case in cases:
                truth = set(case["gold"])
                for m in ARMS:
                    outcomes[m].append(decision(records[case["id"]][m], truth))
            ref = list(zip(*outcomes["top1"]))
            output[str(key)]["seeds"][seed] = {
                arm: {"counts": {k: sum(v) for k, v in zip(METRICS, zip(*outcomes[arm]), strict=True)},
                      "versus_primary": {k: compare(a, b, groups) for k, a, b in
                                        zip(METRICS, zip(*outcomes[arm]), ref, strict=True) if k != "active"}}
                for arm in ARMS}
    return output


def analyze(fixture, reference, measurements, training):
    refs = {r["shard"]: r for r in reference}
    fits = {r["weights_sha256"]: r for r in training}
    assert len(refs) == 4 and len(fits) == 3 and all(r["arm"] == "trained" for r in fits.values())
    labels_sha = measurements[0]["branch_labels_sha256"]
    records = {f["seed"]: {} for f in fits.values()}
    reproduced = Counter()
    provenance = ("fixture_sha256", "corpus_sha256", "models", "templates_sha256", "scoring_labels_sha256", "baseline_sha256")
    for run in measurements:
        seed = fits[run["span_checkpoint_sha256"]]["seed"]
        shard = run["shard"]
        prior = refs[shard]
        assert run["shards"] == prior["shards"] == 4 and run["device"] == "NVIDIA L4"
        assert run["branch_labels_sha256"] == labels_sha and all(run[k] == prior[k] for k in provenance)
        for q, obs, ref in zip(fixture["rows"][shard::4], run["rows"], prior["rows"], strict=True):
            id_, hops = q["id"], q["hops"]
            assert id_ == obs["id"] == ref["id"] and id_ not in records[seed]
            assert set(obs["arms"]) == set(ARMS)
            control = obs["arms"]["top1"]
            assert {k: v for k, v in control.items() if k not in EXTRA} == ref["arms"]["top1"]
            assert control["alternate_final_rank"] is None and not control["alternative_search"]
            reproduced[seed] += 1
            for arm in obs["arms"].values():
                assert all(control[k] == arm[k] for k in control if k not in EXTRA | {"pool"})
                assert control["pool"][:-BUDGETS[hops]] == arm["pool"][:-BUDGETS[hops]]
                selected = arm["fixed_evidence_ids"]
                assert len(selected) == len(set(selected)) == 4 and set(selected) <= set(arm["pool"])
                assert control["fixed_evidence_ids"][:hops-1] == selected[:hops-1]
                if not arm["alternative_search"]:
                    assert arm == control
            records[seed][id_] = obs["arms"]
    assert set(reproduced.values()) == {len(fixture["rows"])} and len(reproduced) == 3
    groups = {h: [q for q in fixture["rows"] if q["hops"] == h] for h in COHORTS}
    return {"fixture_sha256": fixture["fixture_sha256"], "answer_labels_sha256": labels_sha,
            "reference_records": reproduced, "scopes": assess(groups, records)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--baseline", type=Path, nargs=4, required=True)
    p.add_argument("--measurement", type=Path, nargs=12, required=True)
    p.add_argument("--fit", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fitted = [json.loads(f.read_bytes()) for f in a.fit]
    result = analyze(unpack(a.fixture), list(map(unpack, a.baseline)), list(map(unpack, a.measurement)), fitted)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
