#!/usr/bin/env python3
"""Summarize fixed endpoint choices with mirrored orders and relation clusters."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_kam_pretrained_endpoint_selection import BATCH, BOTH_MODES, BOTH_PROTOCOL, MODES, PROTOCOL
from kam_adapter_preservation import ARMS, SEEDS, digest, key
from kam_math_kl_protocol import test_runs, write_json
from summarize_kam_adapter_preservation import read_archive


def family_difference(left, right, families):
    delta = left.astype(np.int8) - right.astype(np.int8)
    groups, index = np.unique(families, return_inverse=True)
    n = np.bincount(index)
    totals = np.bincount(index, weights=delta.mean(axis=0))
    weights = np.full(len(groups), 1 / len(groups))
    draws = np.random.default_rng(731201).multinomial(len(groups), weights, size=20_000)
    samples = (draws @ totals) / (draws @ n)
    return {"mean_delta": float(delta.mean()), "delta_by_seed": delta.mean(axis=1).tolist(),
            "paired_family_bootstrap95_given_seeds": np.quantile(samples, (.025, .975)).tolist()}


def summarize(work, manifest_path, study, model):
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest["protocol"]
    assert protocol in (PROTOCOL, BOTH_PROTOCOL)
    both_earlier = protocol == BOTH_PROTOCOL
    modes = BOTH_MODES if both_earlier else MODES
    assert (tuple(manifest["modes"]), manifest["batch_size"]) == (modes, BATCH)
    selection_path = study / f"{model}-selection.json"
    selection = json.loads(selection_path.read_text())
    solo = read_archive(study / f"{model}-outcomes.json.gz")
    recipe = selection["selected_recipe"]
    runs = test_runs(recipe)
    expected = [(case["pair"], case["family"], q, reverse, i ^ int(reverse))
                for case in manifest["cases"] for i, q in enumerate(case["question_ids"])
                for reverse in (False, True)]
    families = np.array([row[1] for row in expected[::2]])
    cases, counts, hashes = {}, {}, {}
    lengths = None
    for run in runs:
        name = key(*run)
        path = work / f"endpoint-{name}.json"
        data = json.loads(path.read_text())
        meta, rows = data["metadata"], data["records"]
        prior = solo(f"test-{name}")["metadata"]
        assert meta["protocol"] == protocol and (meta["seed"], meta["recipe"], meta["arm"]) == run
        assert meta["manifest_sha256"] == digest(manifest_path)
        assert meta["selection_sha256"] == prior["selection_sha256"] == digest(selection_path)
        for field in ("model", "model_fingerprints", "revision", "gpu", "adapter_sha256", "training"):
            assert meta[field] == prior[field], (name, field)
        saved = [tuple(r[key] for key in ("pair", "family", "question", "reverse", "expected")) for r in rows]
        assert saved == expected and len(rows) == 336
        layout = [tuple(r[m]["tokens"] for m in modes) for r in rows]
        assert all(width[-2] == width[-1] for width in layout)
        lengths = layout if lengths is None else lengths
        assert layout == lengths
        counts[name] = {"seconds": meta["seconds"], "modes": {}}
        has = {}
        incorrect = {}
        for mode in modes:
            for row in rows:
                p = row[mode]
                assert set(p) == {"tokens", "picked", "b_minus_a"} and p["tokens"] > 0
                assert np.isfinite(p["b_minus_a"]) and p["picked"] == int(p["b_minus_a"] > 0)
            indices = np.array([r[mode]["picked"] for r in rows]).reshape((-1, 2))
            gold = np.array([r["expected"] for r in rows]).reshape((-1, 2))
            correct = indices == gold
            has[mode] = correct.all(axis=1)
            incorrect[mode] = (~correct).all(axis=1)
            margins = np.array([r[mode]["b_minus_a"] for r in rows]).reshape((-1, 2))
            signed = margins[:, 0] - margins[:, 1]
            symmetric_correct = (signed != 0) & ((signed > 0) == gold[:, 0])
            has[f"{mode}_symmetric"] = symmetric_correct
            incorrect[f"{mode}_symmetric"] = (signed != 0) & ~symmetric_correct
            counts[name]["modes"][mode] = {
                "presentations_correct": int(correct.sum()), "both_orders_correct": int(has[mode].sum()),
                "order_inconsistent": int((indices[:, 0] == indices[:, 1]).sum()),
                "ties": sum(r[mode]["b_minus_a"] == 0 for r in rows),
                "symmetric_correct": int(symmetric_correct.sum()), "symmetric_ties": int((signed == 0).sum()),
            }
        if both_earlier:
            has["all_orders"] = has["true_first"] & has["true_second"]
            has["both_earlier_symmetric"] = has["true_first_symmetric"] & has["true_second_symmetric"]
        else:
            has["both_orders_counterfactual"] = has["true_prefix"] & incorrect["cross_prefix"]
            has["symmetric_counterfactual"] = has["true_prefix_symmetric"] & incorrect["cross_prefix_symmetric"]
        cases[name] = has
        hashes[name] = digest(path)
    base = cases[key(*runs[-1])]
    fitted = {m: {arm: np.stack([cases[key(seed, recipe, arm)][m] for seed in SEEDS]) for arm in ARMS}
              for m in base}
    results = {
        mode: {"untouched": float(base[mode].mean()),
               "means": {arm: float(v.mean()) for arm, v in arms.items()},
               "kam_minus_dot": family_difference(arms["kam"], arms["dot"], families)}
        for mode, arms in fitted.items()
    }
    before, after = ("true_first", "true_second") if both_earlier else ("endpoints", "true_prefix")
    evidence = {
        "untouched_gain": float(base[after].mean() - base[before].mean()),
        "by_arm": {arm: family_difference(fitted[after][arm], fitted[before][arm], families) for arm in ARMS},
        "change_in_kam_minus_dot": family_difference(
            fitted[after]["kam"].astype(np.int8) + fitted[before]["dot"],
            fitted[after]["dot"].astype(np.int8) + fitted[before]["kam"], families,
        ),
    }
    return {"model": selection["model"], "protocol": protocol, "kl_weight": selection["selected_kl_weight"], "seeds": SEEDS,
            "n_questions": len(families), "relation_families": len(set(families)), "results": results,
            "early_order_effect" if both_earlier else "true_prefix_gain": evidence,
            "checkpoints": counts, "outcome_sha256": hashes,
            "manifest_sha256": digest(manifest_path)}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("manifest", type=Path)
    cli.add_argument("selection_dir", type=Path)
    cli.add_argument("model", choices=("smollm2", "llama3"))
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    write_json(args.output, summarize(args.work, args.manifest, args.selection_dir, args.model))
