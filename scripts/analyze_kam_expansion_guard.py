#!/usr/bin/env python3
"""Fit on development disagreements, lock a guard, and replay held-out pair choices."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analyze_kam_coverage_gate import fit
from analyze_kam_frozen_sequential_oracle import digest, unpack
from analyze_kam_sentence_verifier import paired

ARMS = ("broad", "conditioned_union")
METRICS = ("support", "joint_exact")
BUDGETS = (25, 50, 75, 90, 100)
SHARED = ("index_sha256", "config_sha256", "fitted_sha256", "heads_sha256", "ablations_sha256",
          "baseline_sha256", "models", "indexed_documents", "guard_columns")


def read_rows(path, fixture_path, pools_path, fitted_path, split):
    shards = ([unpack(p) for p in sorted(path.glob(split + "-*.json.gz"))] if path.is_dir() else unpack(path))
    meta = {k: shards[0][k] for k in (*SHARED, "fixture_sha256", "pools_sha256", "split", "shards")}
    assert all(all(s[k] == v for k, v in meta.items()) for s in shards)
    assert sorted(s["shard"] for s in shards) == list(range(meta["shards"]))
    raw = gzip.decompress(fixture_path.read_bytes())
    pools = unpack(pools_path)
    assert meta["fixture_sha256"] == sha256(raw).hexdigest() == pools["fixture_sha256"]
    assert meta["pools_sha256"] == digest(pools_path) and meta["fitted_sha256"] == digest(fitted_path)
    assert meta["index_sha256"] == pools["index_sha256"] and meta["split"] == split
    assert meta["guard_columns"][:2] == ["shared_document", "added_document_count"]
    fixture = json.loads(raw)["rows"][split]
    gold = {q["id"]: set(q["gold_ids"]) for q in fixture}
    rows = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == len(gold) == (3072 if split == "training" else 1024)
    assert {r["id"] for r in rows} == gold.keys()
    for row in rows:
        id_ = row["id"]
        original = pools["rows"][id_]
        pool = set(original)
        assert len(original) == len(pool) == 64 and len(gold[id_]) == 2
        assert row["pool_hash"] == sha256(json.dumps(original, separators=(",", ":")).encode()).hexdigest()
        top = row["corpus_hits"]["conditioned_union"]
        assert row["first"] in pool and row["first"] not in top and len(top) == len(set(top)) == 64
        assert set(row["arms"]) == set(ARMS)
        a, b = (row["arms"][name] for name in ARMS)
        first, second = set(a["pair"]), set(b["pair"])
        assert len(first) == len(second) == 2 and first <= pool and second <= pool | set(top)
        assert a["pair"] != b["pair"] or a["exact"] == b["exact"]
        f = row["features"]
        assert len(f) == len(meta["guard_columns"]) and np.isfinite(f).all()
        assert f[0] == len(first & second) and f[1] == len(second - pool)
        row["complete"] = gold[id_] <= pool
        for arm in row["arms"].values():
            arm["support"] = set(arm["pair"]) == gold[id_]
            arm["joint_exact"] = arm["support"] and arm["exact"]
            assert not arm["support"] or arm["gold_pair_in_shortlist"]
    return rows, meta, shards


def added(row):
    return bool(row["features"][1])


def score(row, models):
    x = np.asarray(row["features"])
    values = [float(np.r_[1, (x - m["mean"]) / m["scale"]] @ m["coefficients"]) for m in models.values()]
    return round(sum(values) / len(values), 7)


def fit_models(rows):
    models = {}
    for metric in METRICS:
        inputs = []
        for row in rows:
            b, u = [row["arms"][arm][metric] for arm in ARMS]
            if added(row) and b != u:
                inputs.append({"variants": {"clean": {"full": u, "without": b, "same": False,
                                                       "features": row["features"]}}})
        model = fit(inputs, ("clean",))
        for key in ("mean", "scale", "coefficients"):
            model[key] = [round(v, 9) for v in model[key]]
        model["union_wins"] = sum(r["variants"]["clean"]["full"] for r in inputs)
        models[metric] = model
    return models


def tally(rows, switches):
    return {m: sum(r["arms"][ARMS[bool(use_union)]][m] for r, use_union in zip(rows, switches, strict=True))
            for m in METRICS}


def lock(rows, meta):
    ordered = sorted(rows, key=lambda r: sha256(("kam-expansion-guard-fit-v1:" + r["id"]).encode()).digest())
    development, calibration = ordered[:2048], ordered[2048:]
    models = fit_models(development)
    ranks = sorted((score(r, models) for r in calibration if added(r)), reverse=True)
    complete = [r for r in calibration if r["complete"]]
    baseline = tally(calibration, [False] * len(calibration))
    base_complete = tally(complete, [False] * len(complete))
    options = {}
    for budget in BUDGETS:
        k = len(ranks) * budget // 100
        selected = next((v for v in reversed(ranks[:k]) if k == len(ranks) or v > ranks[k]), None) if k else None
        chosen = [selected is not None and added(r) and score(r, models) >= selected for r in calibration]
        result = tally(calibration, chosen)
        result_complete = tally(complete, [s for r, s in zip(calibration, chosen, strict=True) if r["complete"]])
        delta = {m: result[m] - baseline[m] for m in METRICS}
        delta_complete = {m: result_complete[m] - base_complete[m] for m in METRICS}
        viable = delta["joint_exact"] >= 2 and all(d >= -(len(complete) // 200) for d in delta_complete.values())
        options[budget] = {"threshold": selected, "switches": sum(chosen), "delta": delta,
                           "complete_delta": delta_complete, "viable": viable}
    best = max((i for i in options if options[i]["viable"]),
               key=lambda i: (options[i]["delta"]["joint_exact"], options[i]["delta"]["support"],
                              -options[i]["switches"]), default=None)
    return {"sources": meta, "models": models, "fit_n": len(development), "calibration_n": len(calibration),
            "calibration_complete_n": len(complete), "eligible_calibration_n": len(ranks),
            "selected_budget_pct": best, "threshold": options[best]["threshold"] if best is not None else None,
            "calibration": options}


def report(rows, policy):
    threshold, models = policy["threshold"], policy["models"]
    choices = {"broad": [False] * len(rows), "conditioned_union": [True] * len(rows),
               "added_only": [added(r) for r in rows],
               "guard": [threshold is not None and added(r) and score(r, models) >= threshold for r in rows]}
    hits = {name: {m: [r["arms"][ARMS[bool(u)]][m] for r, u in zip(rows, uses, strict=True)] for m in METRICS}
            for name, uses in choices.items()}
    contrasts = (("guard", "broad"), ("guard", "conditioned_union"), ("guard", "added_only"),
                 ("conditioned_union", "broad"), ("added_only", "broad"))
    return {"n": len(rows), "switches": {a: sum(u) for a, u in choices.items()},
            "correct": {a: {m: sum(x) for m, x in values.items()} for a, values in hits.items()},
            "oracle": {m: sum(b or u for b, u in zip(hits[ARMS[0]][m], hits[ARMS[1]][m], strict=True)) for m in METRICS},
            "paired": {a + "_vs_" + b: {m: paired(hits[a][m], hits[b][m]) for m in METRICS}
                       for a, b in contrasts} if rows else {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("lock", "evaluate"))
    parser.add_argument("input", type=Path)
    for name in ("fixture", "pools", "fitted", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--write-replay", type=Path)
    a = parser.parse_args()
    rows, meta, shards = read_rows(a.input, a.fixture, a.pools, a.fitted, "training" if a.mode == "lock" else "test")
    if a.mode == "lock":
        result = lock(rows, meta)
    else:
        policy = json.loads(a.policy.read_bytes())
        assert all(meta[k] == policy["sources"][k] for k in SHARED)
        groups = {"all": rows, "complete": [r for r in rows if r["complete"]],
                  "incomplete": [r for r in rows if not r["complete"]]}
        result = {"provenance": meta, "policy_sha256": digest(a.policy),
                  "gpus": dict(sorted(Counter(s["gpu"] for s in shards).items())),
                  "cohorts": {group: report(selected, policy) for group, selected in groups.items()}}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        a.write_replay.write_bytes(gzip.compress(json.dumps(shards, separators=(",", ":"), allow_nan=False).encode(),
                                                 mtime=0))


if __name__ == "__main__":
    main()
