#!/usr/bin/env python3
"""Aggregate seed-paired preservation experiments on the frozen test items."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, MODEL_ID, RECIPES, REVISION, RUNS, SEEDS, TRAIN_RUNS, key
from summarize_kam_general_benchmarks import TASKS, details

BASE = key(*RUNS[-1])


def read_archive(path):
    if path.is_dir():
        def read(name):
            return json.loads((path / f"{name}.json").read_text())
        return read
    with gzip.open(path, "rt") as stream:
        records = json.load(stream)
    return records.__getitem__


def difference(left, right):
    delta = left.astype(np.int8) - right.astype(np.int8)
    deltas_by_seed = delta.mean(axis=1).tolist()
    item_sums = delta.sum(axis=0)
    n = item_sums.size
    support, counts = np.unique(item_sums, return_counts=True)
    samples = np.random.default_rng(731201).multinomial(n, counts / n, size=20000)
    distr = (samples @ support) / (n * delta.shape[0])
    return {"mean_delta": float(delta.mean()), "delta_by_seed": deltas_by_seed,
            "paired_item_bootstrap95_given_seeds": np.quantile(distr, (.025, .975)).tolist()}


def summarize(archive):
    read = read_archive(archive)
    training = {key(*row): read(f"train-{key(*row)}") for row in TRAIN_RUNS}
    comparison = {}
    fingerprints = {run["model_fingerprints"] for run in training.values()}
    assert len(fingerprints) == 1
    assert {run["train_tokens"] for run in training.values()} == {1_048_576}
    assert len({run["new_trainable_parameters"] for run in training.values()}) == 1
    assert len({run["token_manifest_sha256"] for run in training.values()}) == 1
    for seed in SEEDS:
        matched = [training[key(seed, recipe, arm)] for recipe in RECIPES for arm in ARMS]
        assert len({run["exposure_sha256"] for run in matched}) == 1
        assert len({run["gpu"] for run in matched}) == 1
        for recipe, (lr, lam) in RECIPES.items():
            for arm in ARMS:
                run = training[key(seed, recipe, arm)]
                assert (run["seed"], run["recipe"], run["arm"], run["peak_lr"], run["kl_weight"]) == (seed, recipe, arm, lr, lam)
                assert run["training"][-1]["step"] == 256

    for task, n in TASKS.items():
        codes = task in ("humaneval", "mbpp")
        suffix = "-graded" if codes else ""
        payloads = {key(*row): read(f"{task}-{key(*row)}{suffix}") for row in RUNS}
        items = {k: p["records"] for k, p in payloads.items()}
        metadata = {k: p["metadata"] for k, p in payloads.items()}
        for attr in ("source_sha256", "model_fingerprints", "context_tokens", "gpu"):
            assert len({p[attr] for p in metadata.values()}) == 1, (task, attr)
        if codes:
            assert len({p["sandbox_sha256"] for p in metadata.values()}) == 1
        lengths = [r["prompt_tokens"] for r in items[BASE]]
        for seed, recipe, arm in RUNS:
            ident = key(seed, recipe, arm)
            meta, rows = metadata[ident], items[ident]
            assert (meta["task"], meta["seed"], meta["recipe"], meta["arm"]) == (task, seed, recipe, arm)
            assert meta["model"] == MODEL_ID and meta["revision"] == REVISION
            assert meta["model_fingerprints"] in fingerprints
            assert [r["index"] for r in rows] == list(range(n))
            assert [r["prompt_tokens"] for r in rows] == lengths
            if ident != BASE:
                fit = training[ident]
                assert meta["adapter_sha256"] == fit["adapter_sha256"]
                assert all(meta["training"][k] == fit[k] for k in meta["training"])
            else:
                assert meta["adapter_sha256"] is None and meta["new_trainable_parameters"] == 0
        flags = {k: np.asarray([r["correct"] for r in rows]) for k, rows in items.items()}
        groups = {(recipe, arm): np.stack([flags[key(seed, recipe, arm)] for seed in SEEDS])
                  for recipe in RECIPES for arm in ARMS}
        summaries = {}
        for (recipe, arm), rows in groups.items():
            scores = rows.mean(axis=1)
            summaries[f"{recipe}-{arm}"] = {
                "mean": float(scores.mean()), "min": float(scores.min()), "max": float(scores.max()),
                "correct_by_seed": rows.sum(axis=1).tolist(),
                "vs_untouched": difference(rows, flags[BASE][None, :]),
            }
        contrasts = {
            f"{rec}:kam-minus-dot": difference(groups[rec, "kam"], groups[rec, "dot"]) for rec in RECIPES
        }
        for arm in ARMS:
            contrasts[f"{arm}:lower-lr-minus-reference"] = difference(groups["lower_lr", arm], groups["reference", arm])
            contrasts[f"{arm}:kl-minus-lower-lr"] = difference(groups["lower_lr_kl", arm], groups["lower_lr", arm])
            contrasts[f"{arm}:lower-lr-kl-minus-reference"] = difference(groups["lower_lr_kl", arm], groups["reference", arm])
        comparison[task] = {
            "n": n, "source_sha256": metadata[BASE]["source_sha256"],
            "sandbox_sha256": metadata[BASE].get("sandbox_sha256"),
            "gpu": metadata[BASE]["gpu"], "max_prompt_tokens": max(lengths),
            "untouched": details(items[BASE]), "seed_means": summaries, "contrasts": contrasts,
            "per_checkpoint": {k: details(r) for k, r in items.items()},
            "cache_checks": {k: meta["cache_check"] for k, meta in metadata.items()},
        }
    return {"model": MODEL_ID, "revision": REVISION, "seeds": SEEDS, "training": training, "tasks": comparison}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    a.output.write_text(json.dumps(summarize(a.input), indent=2, sort_keys=True) + "\n")
