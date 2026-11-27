#!/usr/bin/env python3
"""Evaluate exact addressing and abstention at matched coverage."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_kam_shared_serving import unpack

ARMS = ("midpoint", "gaussian", "posterior", "event_posterior")
SCENARIOS = ("original", "sham", "intervention")


def evaluate(panels, seed):
    result = {}
    reference = panels["midpoint"][0]
    n = len(reference)
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(n)
    index = rng.integers(0, n, size=(3000, n))
    for arm, (hit, confidence) in panels.items():
        assert len(hit) == len(confidence) == n and np.all((confidence >= 0) & (confidence <= 1))
        order = np.lexsort((tie_breaker, -confidence))
        top = {str(rate): float(hit[order[:int(n * rate)]].mean()) for rate in (0.5, 0.8, 0.95)}
        differences = hit.astype(np.int8) - reference.astype(np.int8)
        interval = np.quantile(differences[index].mean(axis=1), [0.025, 0.975])
        bins = np.minimum((confidence * 10).astype(int), 9)
        error = sum(abs(np.sum((confidence - hit)[bins == b])) for b in range(10)) / n
        result[arm] = {"accuracy": float(hit.mean()), "questions": n, "accuracy_at_coverage": top,
                       "calibration_ece_10": error, "top_label_brier": float(np.mean((confidence - hit)**2)),
                       "paired_delta_vs_midpoint": float(differences.mean()),
                       "paired_95_vs_midpoint": interval.tolist()}
    return result


def generated(record, seed):
    scored = {}
    for n, by_length in record["generated"].items():
        cases = {}
        for condition, modes in by_length["predictions"].items():
            clock = by_length["clock_queries"][condition]
            tests = {"uniform_event": (by_length["gold"], modes),
                     "uniform_clock": (clock["gold"], clock["predictions"])}
            cases[condition] = {}
            for panel, (truth, answers) in tests.items():
                correct = {a: (np.asarray(values["target"]) == truth, np.asarray(values["confidence"]))
                           for a, values in answers.items()}
                cases[condition][panel] = evaluate(correct, seed)
        scored[n] = cases
    return scored


def author(record, seed):
    rows = record["authored"]
    assert len(rows) == len({r["id"] for r in rows}) == 1173
    scored = {}
    for condition in rows[0]["original"]["predictions"]:
        all_modes = {mode: {} for mode in (*SCENARIOS, "both_edits")}
        sham_flips = {}
        for arm in ARMS:
            choices = {}
            for mode in SCENARIOS:
                fields = np.array([r[mode]["predictions"][condition][arm] for r in rows])
                hit = fields[:, 0] == np.array([r[mode]["gold"] for r in rows])
                choices[mode] = fields[:, 0]
                all_modes[mode][arm] = (hit, fields[:, 1])
            original, intervention = (all_modes[mode][arm] for mode in ("original", "intervention"))
            all_modes["both_edits"][arm] = (original[0] & intervention[0], np.minimum(original[1], intervention[1]))
            sham_flips[arm] = float(np.mean(choices["original"] != choices["sham"]))
        metrics = {mode: evaluate(panel, seed) for mode, panel in all_modes.items()}
        # The minimum of two marginal confidences prioritizes pairs for
        # abstention and does not estimate how often both answers are right.
        for statistics in metrics["both_edits"].values():
            del statistics["calibration_ece_10"], statistics["top_label_brier"]
        scored[condition] = {"metrics": metrics, "sham_choice_flip": sham_flips}
    return scored


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs=3, type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    records = sorted(map(unpack, a.runs), key=lambda r: r["seed"])
    assert len({r["fixture_sha256"] for r in records}) == 1
    assert len({r["seed"] for r in records}) == 3
    result = {r["seed"]: {"generated": generated(r, r["seed"]), "authored": author(r, r["seed"])} for r in records}
    a.output.write_text(json.dumps({"fixture_sha256": records[0]["fixture_sha256"], "seeds": result},
                                   indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
