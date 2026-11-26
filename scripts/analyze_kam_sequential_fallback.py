#!/usr/bin/env python3
"""Fit pre-hop fallback models, lock thresholds on calibration, and replay held-out metrics."""

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

METRICS = ("support", "joint_exact")
BUDGETS = (1, 2, 5)


def outcome(row, arm, metric):
    result = row["arms"][arm]
    return result["support"] and (metric == "support" or result["exact"])


def fit_models(rows):
    def train(examples):
        inputs = [{"variants": {"clean": {"full": a, "without": b, "same": False, "features": r["features"]}}}
                  for r, a, b in examples if a != b]
        model = fit(inputs, ("clean",))
        for key in ("mean", "scale", "coefficients"):
            model[key] = [round(v, 9) for v in model[key]]
        return model | {"positive": sum(r["variants"]["clean"]["full"] for r in inputs)}

    models = {"unlinked": train((r, not r["gold_linked"], r["gold_linked"])
                                for r in rows if r["gold_linked"] is not None)}
    for metric in METRICS:
        models[metric] = train((r, outcome(r, "sequential", metric), outcome(r, "broad", metric)) for r in rows)
    return models


def predict(features, models):
    """Score the visible features alone. The first feature marks the selected broad pair's link."""
    x = np.asarray(features)
    logits = {key: round(float(np.r_[1, (x - m["mean"]) / m["scale"]] @ m["coefficients"]), 7)
              for key, m in models.items()}
    return logits | {"fallback": round(sum(logits.values()) / 2, 7)}


def report_policy(rows, use_seq):
    broad = {m: [outcome(r, "broad", m) for r in rows] for m in METRICS}
    selected = {m: [outcome(r, "sequential" if s else "broad", m)
                    for r, s in zip(rows, use_seq, strict=True)] for m in METRICS}
    return {"switches": sum(use_seq), "correct": {m: sum(selected[m]) for m in METRICS},
            "vs_broad": {m: paired(selected[m], broad[m]) for m in METRICS} if rows else {}}


def apply_threshold(row, scores, threshold):
    return threshold is not None and not row["features"][0] and scores[row["id"]]["fallback"] >= threshold


def calibrate(rows, scores):
    eligible = sorted((scores[r["id"]]["fallback"] for r in rows if not r["features"][0]), reverse=True)
    options = {}
    for budget in BUDGETS:
        k = min(len(eligible), len(rows) * budget // 100)
        # Where scores tie at the edge, exclude the entire tied group to honor the budget.
        threshold = eligible[k - 1] if k else None
        if 0 < k < len(eligible) and threshold == eligible[k]:
            threshold = next((s for s in reversed(eligible[:k]) if s > threshold), None)
        report = report_policy(rows, [apply_threshold(r, scores, threshold) for r in rows])
        options[budget] = {"threshold": threshold, **report}
    viable = {budget: r for budget, r in options.items()
              if r["switches"] >= 5 and r["vs_broad"]["joint_exact"]["wins"] - r["vs_broad"]["joint_exact"]["losses"] >= 2
              and r["vs_broad"]["support"]["delta"] >= 0}
    selected = max(viable, key=lambda b: (viable[b]["vs_broad"]["joint_exact"]["delta"],
                                         viable[b]["vs_broad"]["support"]["delta"], -viable[b]["switches"]),
                   default=None)
    return options, selected


def panel(rows, scores, options, selected):
    result = {
        "broad": report_policy(rows, [False] * len(rows)),
        "sequential": report_policy(rows, [True] * len(rows)),
        "broad_pair_unlinked": report_policy(rows, [not r["features"][0] for r in rows]),
    }
    for name, budget in (("conservative", selected), ("five_percent_diagnostic", 5)):
        threshold = options[budget]["threshold"] if budget is not None else None
        result[name] = report_policy(rows, [apply_threshold(r, scores, threshold) for r in rows])
    oracle = {}
    for m in METRICS:
        use = [outcome(r, "sequential", m) and not outcome(r, "broad", m) for r in rows]
        oracle[m] = report_policy(rows, use)["correct"][m]
    return {"n": len(rows), "policies": result, "oracle_correct": oracle}


def auc(rows, labels, scores, field):
    yes = sorted(scores[r["id"]][field] for r, label in zip(rows, labels, strict=True) if label)
    no = sorted(scores[r["id"]][field] for r, label in zip(rows, labels, strict=True) if not label)
    value = None
    if yes and no:
        below = np.searchsorted(no, yes, side="left")
        at_or_below = np.searchsorted(no, yes, side="right")
        value = round(float((below + at_or_below).sum()) / (2 * len(yes) * len(no)), 7)
    return {"positives": len(yes), "negatives": len(no), "auc": value}


def verify_rows(shards, fixture_path, pool_path, fitted_path):
    fields = ("fixture_sha256", "pools_sha256", "index_sha256", "fitted_sha256", "heads_sha256",
              "ablations_sha256", "baseline_sha256", "models", "features", "shards")
    metadata = {k: shards[0][k] for k in fields}
    assert all(all(s[k] == metadata[k] for k in fields) for s in shards)
    assert sorted(s["shard"] for s in shards) == list(range(metadata["shards"]))
    fixture = gzip.decompress(fixture_path.read_bytes())
    sealed = unpack(pool_path)
    assert metadata["fixture_sha256"] == sha256(fixture).hexdigest() == sealed["fixture_sha256"]
    assert metadata["pools_sha256"] == digest(pool_path) and metadata["fitted_sha256"] == digest(fitted_path)
    assert metadata["index_sha256"] == sealed["index_sha256"]
    cases = {q["id"]: (split, q) for split, qs in json.loads(fixture)["rows"].items() for q in qs}
    rows = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == len(cases) == len(sealed["rows"]) == 4096
    assert {r["id"] for r in rows} == cases.keys() == sealed["rows"].keys()
    assert Counter(r["split"] for r in rows) == {"training": 3072, "test": 1024}
    assert len({title for _, q in cases.values() for title in q["gold_titles"]}) == 2 * len(cases)
    assert metadata["features"][:3] == ["broad_pair_link", "pool_link_fraction", "top8_link_fraction"]
    for r in rows:
        split, case = cases[r["id"]]
        ranked = sealed["rows"][r["id"]]
        assert r["split"] == split and len(ranked) == len(set(ranked)) == 64
        assert r["pool_hash"] == sha256(json.dumps(ranked, separators=(",", ":")).encode()).hexdigest()
        assert r["gold_in_pool"] == len(set(ranked) & set(case["gold_ids"]))
        assert (r["gold_linked"] is None) == (r["gold_in_pool"] < 2)
        assert len(r["features"]) == len(metadata["features"]) and np.isfinite(r["features"]).all()
        assert r["features"][0] in (0, 1) and set(r["arms"]) == {"broad", "sequential"}
        assert r["features"][1] == round(r["pool_link_count"] / 2016, 7)
        top8_links = round(r["features"][2] * 28)
        assert 0 <= top8_links <= 28 and r["features"][2] == round(top8_links / 28, 7)
        for arm in r["arms"].values():
            pair = arm["pair"]
            assert len(pair) == len(set(pair)) == 2 and all(0 <= p < 64 for p in pair)
            assert arm["support"] == ({ranked[p] for p in pair} == set(case["gold_ids"]))
        b, s = r["arms"]["broad"], r["arms"]["sequential"]
        assert b["pair"] != s["pair"] or b == s
        if b["support"]:
            assert bool(r["features"][0]) == r["gold_linked"]
    return rows, metadata


def summarize(shards, fixture_path, pool_path, fitted_path):
    rows, metadata = verify_rows(shards, fixture_path, pool_path, fitted_path)
    training = sorted((r for r in rows if r["split"] == "training"),
                      key=lambda r: sha256(("kam-sequential-fallback-fit-v1:" + r["id"]).encode()).digest())
    fit_rows, calibration = training[:2048], training[2048:]
    models = fit_models(fit_rows)
    calibration_scores = {r["id"]: predict(r["features"], models) for r in calibration}
    options, selected = calibrate(calibration, calibration_scores)
    test = [r for r in rows if r["split"] == "test"]
    scores = {r["id"]: predict(r["features"], models) for r in test}
    complete = [r for r in test if r["gold_in_pool"] == 2]
    unlinked = [r for r in complete if r["gold_linked"] is False]
    eligible = [r for r in test if not r["features"][0]]
    cohorts = {"all": test, "gold_in_pool": complete, "gold_unlinked": unlinked,
               "natural_linked_distractor": [r for r in complete if r["gold_linked"] and r["pool_link_count"] > 1],
               "broad_pair_unlinked": eligible, "missing_gold": [r for r in test if r["gold_in_pool"] < 2]}
    diagnostic = {"gold_unlinked_on_complete_pool": auc(complete, [not r["gold_linked"] for r in complete],
                                                       scores, "unlinked")}
    for metric in METRICS:
        for name, cohort in (("visible_eligible", eligible), ("gold_unlinked", unlinked)):
            labels = [outcome(r, "sequential", metric) and not outcome(r, "broad", metric) for r in cohort]
            diagnostic[metric + "_rescue_" + name] = auc(cohort, labels, scores, "fallback")
        diagnostic[metric + "_rescues_vetoed"] = sum(
            bool(r["features"][0]) and outcome(r, "sequential", metric) and not outcome(r, "broad", metric)
            for r in test
        )
    return {**metadata, "gpu": dict(sorted(Counter(s["gpu"] for s in shards).items())),
            "fit": {"n": len(fit_rows), "models": models},
            "calibration": {"n": len(calibration), "selected_budget_pct": selected, "options": options},
            "test": {name: panel(qs, scores, options, selected) for name, qs in cohorts.items()},
            "diagnostics": diagnostic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    for arg in ("fixture", "pools", "fitted", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--write-replay", type=Path)
    a = parser.parse_args()
    shards = [unpack(p) for p in sorted(a.input.glob("score-*.json.gz"))] if a.input.is_dir() else unpack(a.input)
    result = summarize(shards, a.fixture, a.pools, a.fitted)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        payload = json.dumps(shards, separators=(",", ":"), allow_nan=False).encode()
        a.write_replay.write_bytes(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    main()
