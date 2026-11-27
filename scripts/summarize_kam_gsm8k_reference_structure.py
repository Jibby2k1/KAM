#!/usr/bin/env python3
"""Summarize frozen GSM8K outcomes by reference calculator annotations."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, LOW_RATE_KL_RECIPE, SEEDS, key
from summarize_kam_adapter_preservation import BASE, difference, read_archive


def marker_balance(lhs, rhs, lhs_parsed, rhs_parsed):
    change = lhs.astype(np.int8) - rhs.astype(np.int8)
    conditions = {
        "both_marked": lhs_parsed & rhs_parsed,
        "left_only_marked": lhs_parsed & ~rhs_parsed,
        "right_only_marked": ~lhs_parsed & rhs_parsed,
        "neither_marked": ~lhs_parsed & ~rhs_parsed,
    }
    return {name: {"n_by_seed": cond.sum(axis=1).tolist(),
                   "net_correct_change_by_seed": (change * cond).sum(axis=1).tolist()}
            for name, cond in conditions.items()}


def summarize(structure, primary, calibrated, pilot=None):
    base = primary(f"gsm8k-{BASE}")
    assert structure["source_sha256"] == base["metadata"]["source_sha256"]
    rows = structure["rows"]
    assert [r["index"] for r in rows] == list(range(len(base["records"])))
    steps = np.array([r["calculator_annotations"] for r in rows])
    division = np.array([r["reference_division"] for r in rows])
    groups = {"all": np.ones(len(steps), dtype=bool),
              "0-1": steps <= 1, "2": steps == 2, "3": steps == 3, "4+": steps >= 4,
              "division": division, "no_division": ~division}
    # Every condition uses the same fixture order and the same three seeds.
    stages = {"untouched": [base]}
    for recipe, source in (("lower_lr", primary), ("lower_lr_kl", primary), (LOW_RATE_KL_RECIPE, pilot)):
        if source is not None:
            for arm in ARMS:
                stages[f"{recipe}-{arm}"] = [source(f"gsm8k-{key(seed, recipe, arm)}") for seed in SEEDS]
    stages["lower_lr-dot-scaled"] = [calibrated[f"low-rate-matched-gsm8k-{seed}"] for seed in SEEDS]
    values, parsed = {}, {}
    for tag, samples in stages.items():
        for payload in samples:
            assert payload["metadata"]["source_sha256"] == structure["source_sha256"]
            assert [r["index"] for r in payload["records"]] == list(range(len(steps)))
        values[tag] = np.array([[r["correct"] for r in x["records"]] for x in samples])
        parsed[tag] = np.array([[r["parsed"] for r in x["records"]] for x in samples])
        assert (values[tag] <= parsed[tag]).all()

    summary = {}
    for tag, mask in groups.items():
        scores = {condition: {"score": float(values[condition][:, mask].mean()),
                              "marker_rate": float(parsed[condition][:, mask].mean()),
                              "correct_by_seed": values[condition][:, mask].sum(axis=1).tolist()}
                  for condition in values}
        comparisons = {"kam-minus-untouched": ("lower_lr-kam", "untouched"),
                       "dot-minus-untouched": ("lower_lr-dot", "untouched"),
                       "scaled-minus-unscaled-dot": ("lower_lr-dot-scaled", "lower_lr-dot"),
                       "scaled-dot-minus-kam": ("lower_lr-dot-scaled", "lower_lr-kam")}
        if pilot:
            for arm in ARMS:
                comparisons[f"intermediate-kl-effect:{arm}"] = (f"{LOW_RATE_KL_RECIPE}-{arm}", f"lower_lr-{arm}")
        summary[tag] = {
            "n": int(mask.sum()), "condition_scores": scores,
            "contrasts": {name: difference(values[a][:, mask], values[b][:, mask])
                          for name, (a, b) in comparisons.items()},
            "correctness_changes_by_marker": {
                name: marker_balance(values[a][:, mask], values[b][:, mask], parsed[a][:, mask], parsed[b][:, mask])
                for name, (a, b) in comparisons.items()
            },
        }
    return {"model": base["metadata"]["model"], "source_sha256": structure["source_sha256"],
            "n": len(steps), "groups": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path)
    parser.add_argument("primary", type=Path)
    parser.add_argument("calibrated", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    structure = json.loads(args.structure.read_text())
    primary = read_archive(args.primary)
    with gzip.open(args.calibrated, "rt") as stream:
        raw = json.load(stream)
    by_model = next(group for group in raw.values()
                    if next(iter(group.values()))["metadata"]["model"] == primary(f"gsm8k-{BASE}")["metadata"]["model"])
    pilot = read_archive(args.pilot) if args.pilot else None
    result = summarize(structure, primary, by_model, pilot)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
