#!/usr/bin/env python3
"""Standardize MATH-500 answer-type contrasts to common subject-by-level cells."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, key
from kam_math_kl_protocol import write_json
from summarize_kam_adapter_preservation import read_archive

RESAMPLES = 20_000


def cohort_draws(values, rng):
    sums = values.sum(axis=0)
    support, counts = np.unique(sums, return_counts=True)
    sampled = rng.multinomial(sums.size, counts / sums.size, size=RESAMPLES)
    return (sampled @ support) / values.size


def adjusted(archive, selection, integer_manifest):
    original = read_archive(archive)
    locked = json.loads(selection.read_text())
    recipe = locked["selected_recipe"]
    integer_ids = set(json.loads(integer_manifest.read_text())["fixtures"]["test"]["ids"])
    by_arm = {arm: [original(f"graded-{key(seed, recipe, arm)}")["records"] for seed in SEEDS] for arm in ARMS}
    reference = by_arm["kam"][0]
    ids = [row["id"] for row in reference]
    assert len(ids) == 500 and len(set(ids) & integer_ids) == 311
    assert all([row["id"] for row in cohort] == ids for group in by_arm.values() for cohort in group)
    flags = {arm: np.array([[row["full"]["correct"] for row in cohort] for cohort in group], dtype=np.int8)
             for arm, group in by_arm.items()}
    delta = flags["kam"] - flags["dot"]
    cells = defaultdict(lambda: ([], []))
    for i, row in enumerate(reference):
        cells[row["subject"], row["level"]][int(row["id"] not in integer_ids)].append(i)
    rng = np.random.default_rng(731201)
    total_weight = 0
    points = np.zeros((len(SEEDS), 2))
    draws = np.zeros((RESAMPLES, 2))
    weights = []
    for (subject, level), (integer, noninteger) in sorted(cells.items()):
        weight = min(len(integer), len(noninteger))
        weights.append({"subject": subject, "level": level,
                        "integer": len(integer), "noninteger": len(noninteger), "weight": weight})
        if not weight:
            continue
        total_weight += weight
        for i, group in enumerate((integer, noninteger)):
            values = delta[:, group]
            points[:, i] += weight * values.mean(axis=1)
            draws[:, i] += weight * cohort_draws(values, rng)
    points /= total_weight
    draws /= total_weight
    label = ("integer", "noninteger", "noninteger-minus-integer")
    points = np.column_stack((points, points[:, 1] - points[:, 0]))
    draws = np.column_stack((draws, draws[:, 1] - draws[:, 0]))
    effects = {kind: {"mean_delta": float(points[:, i].mean()), "delta_by_seed": points[:, i].tolist(),
                      "paired_item_bootstrap95_given_seeds": np.quantile(draws[:, i], (.025, .975)).tolist()}
               for i, kind in enumerate(label)}
    return {"model": locked["model"], "seeds": SEEDS, "common_support_weight": total_weight,
            "cells": weights, "standardized_kam_minus_dot": effects}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("selection", type=Path)
    cli.add_argument("integer_manifest", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    write_json(args.output, adjusted(args.archive, args.selection, args.integer_manifest))
