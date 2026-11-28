#!/usr/bin/env python3
"""Count persistent math-answer changes from the verified, numeric outcome archives."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, digest, key
from kam_math_kl_protocol import test_runs, write_json
from summarize_kam_adapter_preservation import read_archive


def analyze(archive, selection_path):
    selection = json.loads(selection_path.read_text())
    recipe = selection["selected_recipe"]
    read = read_archive(archive)
    runs = test_runs(recipe)
    outcomes = {run: read(f"graded-{key(*run)}") for run in runs}
    untouched = outcomes[runs[-1]]["records"]
    ids = [row["id"] for row in untouched]
    for (seed, run_recipe, arm), output in outcomes.items():
        metadata = output["metadata"]
        assert (metadata["seed"], metadata["recipe"], metadata["arm"]) == (seed, run_recipe, arm)
        assert metadata["selection_sha256"] == digest(selection_path)
        assert [row["id"] for row in output["records"]] == ids

    base = np.array([row["full"]["correct"] for row in untouched])
    fits = [outcomes[run]["records"] for run in runs[:-1]]
    correct = np.array([[row["full"]["correct"] for row in rows] for rows in fits])
    parsed = np.array([[row["full"]["parsed"] for row in rows] for rows in fits])
    capped = np.array([[row["stop"] == "budget" for row in rows] for rows in fits])
    run_arms = np.array([arm for _, _, arm in runs[:-1]])
    groups = {arm: summarize(base, correct[run_arms == arm], parsed[run_arms == arm],
                             capped[run_arms == arm]) for arm in ARMS}
    groups["both_arms"] = summarize(base, correct, parsed, capped)
    disagreements = correct[run_arms == "kam"] != correct[run_arms == "dot"]
    return {"model": selection["model"], "seeds": SEEDS, "n": len(ids),
            "untouched_correct": int(base.sum()), "groups": groups,
            "arm_disagreements_by_seed": disagreements.sum(axis=1).tolist(),
            "archive_sha256": digest(archive), "selection_sha256": digest(selection_path)}


def summarize(base, correct, parsed, capped):
    losses = base & ~correct
    gains = ~base & correct
    persistent = losses.all(axis=0)
    return {
        "fitted_checkpoints": len(correct),
        "lost_count_by_checkpoint": losses.sum(axis=1).tolist(),
        "gained_count_by_checkpoint": gains.sum(axis=1).tolist(),
        "lost_in_every_fit": int(persistent.sum()),
        "lost_in_every_fit_all_parsed": int((persistent & parsed.all(axis=0)).sum()),
        "lost_in_every_fit_no_cap_hits": int((persistent & ~capped.any(axis=0)).sum()),
        "lost_in_every_fit_verified_extract_and_no_cap_hits": int(
            (persistent & parsed.all(axis=0) & ~capped.any(axis=0)).sum()
        ),
        "gained_in_every_fit": int(gains.all(axis=0).sum()),
        "lost_in_any_fit": int(losses.any(axis=0).sum()),
        "gained_in_any_fit": int(gains.any(axis=0).sum()),
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("selection", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    write_json(args.output, analyze(args.archive, args.selection))
