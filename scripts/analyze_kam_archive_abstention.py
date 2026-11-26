#!/usr/bin/env python3
"""Calibrate revision confidence on the Philadelphia Fed and replay Dallas transfer."""

import argparse
from collections import defaultdict
import gzip
from itertools import groupby
import json
from math import ceil
from pathlib import Path
from statistics import mean, median

from analyze_kam_oecd_revisions import ARMS

SCORE_COLUMNS = {"payload_max_probability": 5, "payload_margin": 6}


def calibrated_threshold(rows, column):
    known = sorted(r[column] for r in rows if r[4])
    return known[len(known) - ceil(.9 * len(known))]


def partition(rows, threshold):
    for row in rows:
        _, _, _, _, known, probability, margin, valid, correct = row
        choices = {
            "always_emit": True,
            "payload_max_probability": probability >= threshold["payload_max_probability"],
            "payload_margin": margin >= threshold["payload_margin"],
            "exact_eligibility_check": valid,
        }
        for method, accepted in choices.items():
            yield method, known, accepted, correct


def metrics(decisions):
    known_total = sum(known for known, _, _ in decisions)
    unknown_total = len(decisions) - known_total
    known_accept = sum(known and accepted for known, accepted, _ in decisions)
    unknown_accept = sum(not known and accepted for known, accepted, _ in decisions)
    exact_accepted = sum(known and accepted and correct for known, accepted, correct in decisions)
    return {"answerable": known_total, "unavailable": unknown_total,
            "answerable_accepted": known_accept, "unavailable_accepted": unknown_accept,
            "answerable_correct_accepted": exact_accepted}


def availability_ranking(rows, column):
    # Compare all answerable/unavailable pairs within a checkpoint, giving ties half credit.
    ordered = sorted((r[column], r[4]) for r in rows)
    negatives, concordance = 0, 0.
    for _, block in groupby(ordered, key=lambda row: row[0]):
        labels = [known for _, known in block]
        n_known = sum(labels)
        n_missing = len(labels) - n_known
        concordance += n_known * (negatives + .5 * n_missing)
        negatives += n_missing
    n_known = len(ordered) - negatives
    paired = {(r[0], r[2], r[4]): r[column] for r in rows}
    assert n_known == negatives == len(paired) // 2
    strictly_higher, ties = 0, 0
    for (id_, recode, known), value in paired.items():
        if known:
            missing = paired[id_, recode, False]
            strictly_higher += value > missing
            ties += value == missing
    return {"auroc": concordance / (n_known * negatives), "known_higher_pairs": strictly_higher,
            "tied_pairs": ties, "pairs": n_known}


def summarize(runs):
    assert len(runs) == 8
    assert len({r["gpu"] for r in runs}) == 1
    assert len({r["seed"] for r in runs}) == 4
    assert len({(r["arm"], r["seed"]) for r in runs}) == 8
    assert {r["train_tokens"] for r in runs} == {13516800}
    assert len({r["calibration_sha256"] for r in runs}) == len({r["test_sha256"] for r in runs}) == 1
    assert len({tuple(r["train_input_sha256"]) for r in runs}) == 4
    for seed in {r["seed"] for r in runs}:
        assert len({tuple(r["train_input_sha256"]) for r in runs if r["seed"] == seed}) == 1
    thresholds = defaultdict(list)
    raw = defaultdict(list)
    ranks = defaultdict(list)
    for run in runs:
        arm = run["arm"]
        assert arm in ARMS
        assert len(run["calibration"]) == 160 * 4 and len(run["test"]) == 292 * 8
        assert len({tuple(r[:5]) for r in run["calibration"]}) == len(run["calibration"])
        assert len({tuple(r[:5]) for r in run["test"]}) == len(run["test"])
        fit = {method: calibrated_threshold(run["calibration"], column) for method, column in SCORE_COLUMNS.items()}
        for method, value in fit.items():
            thresholds[arm, method].append(value)
        for condition, rows in (("Philadelphia_calibration", run["calibration"]),
                                ("Dallas_observed", [r for r in run["test"] if r[3] == 1]),
                                ("Dallas_clock_tripled", [r for r in run["test"] if r[3] == 3])):
            assert all((not r[-1] or r[-2]) and (r[4] or not r[-2]) for r in rows)
            for method, known, accepted, correct in partition(rows, fit):
                raw[arm, condition, method].append((known, accepted, correct))
            for name, column in SCORE_COLUMNS.items():
                ranks[arm, condition, name].append(availability_ranking(rows, column))
    return {"calibration_sha256": runs[0]["calibration_sha256"], "test_sha256": runs[0]["test_sha256"],
            "gpu": runs[0]["gpu"], "target_answerable_acceptance": .9,
            "thresholds": {f"{arm}:{name}": {"min": min(values), "median": median(values), "max": max(values)}
                           for (arm, name), values in sorted(thresholds.items())},
            "confidence_ranking": {f"{arm}:{condition}:{method}":
                                   {"mean_auroc": mean(v["auroc"] for v in values)}
                                   | {key: sum(v[key] for v in values)
                                      for key in ("known_higher_pairs", "tied_pairs", "pairs")}
                                   for (arm, condition, method), values in sorted(ranks.items())},
            "metrics": {f"{arm}:{condition}:{method}": metrics(decisions)
                        for (arm, condition, method), decisions in sorted(raw.items())}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--write-replay", type=Path)
    a = p.parse_args()
    if a.input.is_dir():
        runs = [json.loads(p.read_text()) for p in sorted(a.input.glob("confidence-*.json"))]
    else:
        runs = json.loads(gzip.decompress(a.input.read_bytes()))
    a.output.write_text(json.dumps(summarize(runs), indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        a.write_replay.write_bytes(gzip.compress(json.dumps(runs, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
