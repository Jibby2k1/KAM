#!/usr/bin/env python3
"""Select a shared KL weight on development and score only that weight on test."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, LOW_RATE_KL_RECIPE, MODEL_REGISTRY, SEEDS, TRAINING_RECIPES, digest, key
from kam_math_kl_protocol import BUDGET, DEV_RUNS, MARKERS, PROTOCOL, RECIPES, test_runs, write_json
from summarize_kam_adapter_preservation import difference, read_archive
from summarize_kam_gsm8k_reference_structure import marker_balance

MODES = (*MARKERS, "both")


def load_split(work, manifest_path, split, runs, model, factorial, pilot, selection_sha=None):
    manifest = json.loads(manifest_path.read_text())
    fixture = manifest["fixtures"][split]
    assert manifest["protocol"] == PROTOCOL and manifest["recipes_in_tie_break_order"] == list(RECIPES)
    assert manifest["markers"] == MARKERS and manifest["budget"] == BUDGET
    paths = {}
    results = {}
    prompt_lengths = None
    for seed, recipe, arm in runs:
        ident = key(seed, recipe, arm)
        original = (pilot if recipe == LOW_RATE_KL_RECIPE else factorial)(f"gsm8k-{ident}")["metadata"]
        path = work / f"{split}-{ident}.json"
        result = json.loads(path.read_text())
        metadata, rows = result["metadata"], result["records"]
        assert [r["id"] for r in rows] == fixture["ids"] and len(rows) == fixture["n"]
        expected = {"protocol": PROTOCOL, "model": model, "seed": seed, "recipe": recipe, "arm": arm,
                    "split": split, "fixture_sha256": fixture["sha256"], "selection_sha256": selection_sha,
                    "manifest_sha256": digest(manifest_path)}
        assert all(metadata[name] == value for name, value in expected.items()), ident
        for name in ("revision", "model_fingerprints", "adapter_sha256", "gpu", "new_trainable_parameters",
                     "training"):
            assert metadata[name] == original[name], (ident, name)
        for r in rows:
            assert set(r) == {"id", *MARKERS}
            for label in MARKERS:
                cell = r[label]
                assert set(cell) == {"correct", "parsed", "stop", "prompt_tokens", "output_tokens"}
                assert type(cell["correct"]) is type(cell["parsed"]) is bool
                assert cell["correct"] <= cell["parsed"] and (cell["stop"] != "pattern" or cell["parsed"])
                assert 0 <= cell["output_tokens"] <= BUDGET and cell["prompt_tokens"] > 0
                assert cell["stop"] in ("pattern", "eos", "budget")
        lengths = [tuple(row[label]["prompt_tokens"] for label in MARKERS) for row in rows]
        if prompt_lengths is None:
            prompt_lengths = lengths
        assert lengths == prompt_lengths, ident
        results[ident] = result
        paths[ident] = digest(path)
    return results, paths


def vectors(payloads, flag):
    flags = {}
    for ident, result in payloads.items():
        cell = {m: np.array([row[m][flag] for row in result["records"]]) for m in MARKERS}
        cell["both"] = np.logical_and(cell["hash"], cell["answer"])
        flags[ident] = cell
    return flags


def report_runs(payloads, flags):
    return {
        ident: {
            "correct": {mode: int(values.sum()) for mode, values in flags[ident].items()},
            "parsed": {m: sum(row[m]["parsed"] for row in result["records"]) for m in MARKERS},
            "stops": {m: dict(Counter(row[m]["stop"] for row in result["records"])) for m in MARKERS},
            "seconds": result["metadata"]["seconds"],
        }
        for ident, result in payloads.items()
    }


def grouped(flags, recipe, arm):
    return {mode: np.stack([flags[key(seed, recipe, arm)][mode] for seed in SEEDS]) for mode in MODES}


def select(work, manifest, model, factorial, pilot):
    payloads, paths = load_split(work, manifest, "dev", DEV_RUNS, model, factorial, pilot)
    flags = vectors(payloads, "correct")
    n = len(next(iter(payloads.values()))["records"])
    candidates = {}
    for recipe in RECIPES:
        arms = {arm: grouped(flags, recipe, arm) for arm in ARMS}
        candidates[recipe] = {
            "kl_weight": TRAINING_RECIPES[recipe][1],
            "mean_exact": {arm: {mode: float(values.mean()) for mode, values in result.items()}
                           for arm, result in arms.items()},
            "pooled_both_correct": int(sum(result["both"].sum() for result in arms.values())),
        }
    winner = max(RECIPES, key=lambda recipe: candidates[recipe]["pooled_both_correct"])
    return {"protocol": PROTOCOL, "model": model, "selected_recipe": winner,
            "selected_kl_weight": TRAINING_RECIPES[winner][1], "manifest_sha256": digest(manifest),
            "n": n, "seeds": SEEDS, "candidate_results": candidates, "outcome_sha256": paths,
            "checkpoints": report_runs(payloads, flags)}


def summarize(work, manifest, model, factorial, pilot, selection_path):
    selection = json.loads(selection_path.read_text())
    assert selection == json.loads(json.dumps(select(work, manifest, model, factorial, pilot)))
    recipe = selection["selected_recipe"]
    runs = test_runs(recipe)
    payloads, paths = load_split(work, manifest, "test", runs, model, factorial, pilot, digest(selection_path))
    flags = vectors(payloads, "correct")
    base = flags[key(*runs[-1])]
    arms = {arm: grouped(flags, recipe, arm) for arm in ARMS}
    parse_flags = vectors(payloads, "parsed")
    parsed = {arm: grouped(parse_flags, recipe, arm) for arm in ARMS}
    contrasts = {}
    for mode in MODES:
        contrasts[f"{mode}:kam-minus-dot"] = difference(arms["kam"][mode], arms["dot"][mode])
        for arm, values in arms.items():
            contrasts[f"{mode}:{arm}-minus-untouched"] = difference(values[mode], base[mode][None, :])
    contrasts["answer-vs-hash:change-in-kam-minus-dot"] = difference(
        arms["kam"]["answer"].astype(np.int8) + arms["dot"]["hash"],
        arms["dot"]["answer"].astype(np.int8) + arms["kam"]["hash"],
    )
    return {"protocol": PROTOCOL, "model": model, "n": len(base["both"]), "seeds": SEEDS,
            "selected_recipe": recipe, "selected_kl_weight": selection["selected_kl_weight"],
            "selection_sha256": digest(selection_path), "outcome_sha256": paths,
            "untouched": {mode: float(values.mean()) for mode, values in base.items()},
            "mean_exact": {arm: {mode: float(values.mean()) for mode, values in result.items()}
                           for arm, result in arms.items()},
            "contrasts": contrasts, "checkpoints": report_runs(payloads, flags),
            "correctness_changes_by_marker": {
                mode: marker_balance(arms["kam"][mode], arms["dot"][mode], parsed["kam"][mode], parsed["dot"][mode])
                for mode in MODES
            }}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("manifest", type=Path)
    cli.add_argument("model", choices=MODEL_REGISTRY)
    cli.add_argument("factorial", type=Path)
    cli.add_argument("pilot", type=Path)
    cli.add_argument("selection", type=Path)
    cli.add_argument("--test-output", type=Path)
    args = cli.parse_args()
    data = (args.work, args.manifest, args.model, read_archive(args.factorial), read_archive(args.pilot))
    if args.test_output is None:
        write_json(args.selection, select(*data))
    else:
        write_json(args.test_output, summarize(*data, args.selection))
