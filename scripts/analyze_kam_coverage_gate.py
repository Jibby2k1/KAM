#!/usr/bin/env python3
"""Fit a paired coverage policy on the historical cohorts and score frozen tests."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from analyze_kam_fullwiki_support_links import paired

VARIANTS = ("clean", "sham", "linked")


def fit(rows, variants):
    cases = []
    labels = []
    costs = []
    for row in rows:
        disagree = [v for name, v in row["variants"].items() if name in variants
                    and v["full"] != v["without"]]
        for entry in disagree:
            assert not entry["same"]
            cases.append(entry["features"])
            labels.append(entry["full"])
            costs.append(1 / len(disagree))
    x = np.asarray(cases)
    y = np.asarray(labels, dtype=float)
    weight = np.asarray(costs)
    weight /= weight.sum()
    center = np.average(x, axis=0, weights=weight)
    scale = np.sqrt(np.average((x-center) ** 2, axis=0, weights=weight)).clip(.001)
    x = np.column_stack((np.ones(len(x)), (x-center)/scale))
    penalty = np.array([0.0001] + [.05] * (x.shape[1] - 1))
    beta = np.zeros(x.shape[1])
    for _ in range(20):
        p = 1/(1 + np.exp(-np.clip(x @ beta, -35, 35)))
        gradient = x.T @ (weight*(p-y)) + penalty*beta
        jacobian = (x.T * (weight*p*(1-p))) @ x + np.diag(penalty)
        step = np.linalg.solve(jacobian, gradient)
        beta -= step
        if np.linalg.norm(step) < 1e-9:
            break
    assert np.all(np.isfinite(beta))
    return {"mean": center.tolist(), "scale": scale.tolist(), "coefficients": beta.tolist(),
            "training_disagreements": len(x)}


def choose(entry, fitted):
    x = (np.asarray(entry["features"]) - fitted["mean"]) / fitted["scale"]
    return "full" if float(np.r_[1, x] @ fitted["coefficients"]) >= 0 else "without"


def score(rows, models):
    result = {}
    for variant in VARIANTS:
        selected = [r["variants"][variant] for r in rows if variant in r["variants"]]
        keys = ("full", "without", "candidate_link_rule", "learned", "learned_clean", "oracle")
        outcome = {key: [] for key in keys}
        full_selections = Counter()
        for v in selected:
            routed = {"learned": choose(v, models["paired"]), "learned_clean": choose(v, models["clean"]),
                      "candidate_link_rule": "without" if v["features"][1] else "full"}
            for key in keys:
                outcome[key].append((v["full"] or v["without"]) if key == "oracle" else v[routed.get(key, key)])
            full_selections.update(n for n, arm in routed.items() if arm == "full")
        result[variant] = {"n": len(selected), "correct": {key: sum(values) for key, values in outcome.items()},
                           "full_selections": dict(full_selections),
                           "same_pair": sum(v["same"] for v in selected),
                           "learned_vs_baseline": {key: {base: paired([int(a)-int(b) for a,b in
                                                                         zip(outcome[key], outcome[base], strict=True)])
                                                         for base in ("full", "without", "candidate_link_rule")}
                                                    for key in ("learned", "learned_clean")}}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    shards = [json.loads(f.read_text()) for f in sorted(a.directory.glob("*-?.json"))]
    assert len(shards) == 12 and Counter(s["dataset"] for s in shards) == {"twowiki": 8, "musique": 4}
    for key in ("fixture_sha256", "split_sha256", "heads_sha256", "ablations_sha256", "gate_columns"):
        assert all(s[key] == shards[0][key] for s in shards)
    rows = [q for s in shards for q in s["rows"]]
    assert len(rows) == len({r["id"] for r in rows})
    counts = dict(Counter(r["split"] for r in rows))
    assert counts == {"twowiki-fit": 768, "musique-fit": 236, "twowiki-test": 420, "musique-test": 300}
    for r in rows:
        variants = r["variants"]
        assert tuple(variants) in (("clean",), VARIANTS)
        assert all(len(v["features"]) == len(shards[0]["gate_columns"]) for v in variants.values())
    train = [r for r in rows if r["split"].endswith("-fit")]
    models = {"paired": fit(train, VARIANTS), "clean": fit(train, ("clean",))}
    result = {key: shards[0][key] for key in ("fixture_sha256", "split_sha256", "heads_sha256", "ablations_sha256",
                                            "gate_columns")}
    result.update({"split_counts": counts, "models": models, "tests": {
        name: score([r for r in rows if r["split"] == name], models) for name in sorted(counts) if name.endswith("-test")}})
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
