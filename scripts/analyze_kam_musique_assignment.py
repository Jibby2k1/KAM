#!/usr/bin/env python3
"""Freeze and audit distinct final paragraph assignments across learned subqueries."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

METHODS = ("highest_affinity", "greedy_subquery", "joint_assignment", "direct_path")


def digest(file):
    return sha256(file.read_bytes()).hexdigest()


def evidence(row, old_path):
    pool = row["pool"]
    assert len(pool) == len(set(pool)) == 128
    if row["fallback"]:
        assert pool[:4] == old_path
        return {m: old_path for m in METHODS}, False
    values = np.asarray(row["subquery_affinity"], dtype=np.float64)
    k = row["predicted_steps"]
    assert values.shape == (k, 128) and k in (2, 3, 4)
    scale = values.std(axis=1, keepdims=True).clip(min=1e-8)
    z = (values - values.mean(1, keepdims=True)) / scale
    max_order = np.argsort(-z.max(0), kind="stable").tolist()
    greedy = []
    for step_values in z:
        order = np.argsort(-step_values, kind="stable")
        greedy.append(next(i for i in order if i not in greedy))
    _, coupled = linear_sum_assignment(z, maximize=True)
    joint = coupled.tolist()
    assert len(greedy) == len(set(greedy)) == len(joint) == len(set(joint)) == k
    max_greedy = z[np.arange(k), greedy].sum()
    max_joint = z[np.arange(k), joint].sum()
    assert max_joint + 1e-10 >= max_greedy
    result = {"highest_affinity": [pool[i] for i in max_order[:4]], "direct_path": old_path}
    for name, candidates in (("greedy_subquery", greedy), ("joint_assignment", joint)):
        selected = list(dict.fromkeys(candidates + max_order))[:4]
        result[name] = [pool[i] for i in selected]
    return result, joint != greedy


def assess(cases, selected):
    cluster = shared_evidence_clusters(cases)
    results = []
    for index in range(3):
        metrics = {a: Counter() for a in METHODS}
        flags = {a: [] for a in METHODS}
        changed = 0
        for q in cases:
            docs, differs = selected[q["id"]][index]
            gold = set(q["gold"])
            changed += differs
            for method, found in docs.items():
                assert len(found) == len(set(found)) == 4
                n = len(gold.intersection(found))
                success = n == len(gold)
                metrics[method].update({"all_support": success, "gold_paragraphs": n})
                flags[method].append(success)
        results.append({"metrics": metrics, "assigned_differently_to_greedy": changed,
                        "paired": {m: {b: compare(flags[m], flags[b], cluster) for b in ("direct_path", "greedy_subquery")}
                                   for m in ("highest_affinity", "joint_assignment")}})
    return {"questions": len(cases), "seeds": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "write-policy", "write-picks"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("affinity", "old"):
        p.add_argument("--" + name, action="append", nargs=4, type=Path, required=True)
    p.add_argument("--reader", type=Path, nargs=4, required=True)
    a = p.parse_args()
    f = unpack(a.manifest)
    old_prompt = sorted(map(unpack, a.reader), key=lambda c: c["shard"])
    assert len(a.affinity) == len(a.old) == 3
    metadata = {"fixture_sha256": f["fixture_sha256"], "corpus_sha256": f["corpus_sha256"]}
    selected, weights = {}, []
    for i, (fresh, baseline) in enumerate(zip(a.affinity, a.old, strict=True)):
        new, old = [sorted(map(unpack, p), key=lambda c: c["shard"]) for p in (fresh, baseline)]
        weights.append(old[0]["weights_sha256"])
        for n, re, reader in zip(new, old, old_prompt, strict=True):
            assert n["shard"] == re["shard"] == reader["shard"]
            assert all(n[k] == re[k] == v for k, v in metadata.items())
            assert n["weights_sha256"] == re["weights_sha256"] == weights[-1]
            assert n["generation_sha256"] == re["generation_sha256"] and n["device"] == "NVIDIA L4"
            for now, previous, prompt in zip(n["rows"], re["rows"], reader["rows"], strict=True):
                id_ = now["id"]
                assert id_ == previous["id"] == prompt["id"]
                assert {k: v for k, v in now.items() if k != "subquery_affinity"} == previous
                before = prompt["arms"][f"learned_path/{i}"]["evidence_ids"]
                selection = evidence(now, before)
                selected.setdefault(id_, []).append(selection)
    cases = f["rows"]
    assert set(selected) == {q["id"] for q in cases} and all(len(v) == 3 for v in selected.values())
    scope = {"all": cases, **{str(k): [q for q in cases if q["hops"] == k] for k in COHORTS}}
    policy = {"method": "joint_assignment", "parameters": {"standardize_per_subquery": True, "fill": "highest_affinity"},
              "test_fixture_sha256": f["fixture_sha256"], "affinity_sha256": [list(map(digest, s)) for s in a.affinity]}
    a.write_policy.write_text(json.dumps(policy, separators=(",", ":")) + "\n")
    picks = metadata | {"policy_sha256": digest(a.write_policy), "provenance": metadata | {"weights_sha256": weights},
                        "rows": [{"id": q["id"], "selections": [r[0]["joint_assignment"] for r in selected[q["id"]]]}
                                 for q in cases]}
    a.write_picks.write_text(json.dumps(picks, separators=(",", ":")) + "\n")
    result = metadata | {"scopes": {k: assess(q, selected) for k, q in scope.items()}}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
