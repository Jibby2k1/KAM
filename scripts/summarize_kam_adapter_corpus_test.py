#!/usr/bin/env python3
"""Summarize held-out continuation loss and perplexity by corpus domain."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, FACTORIAL_RECIPES, SEEDS, key
from summarize_kam_adapter_preservation import BASE


def contrast(values, weights):
    per_chunk = values.mean(axis=0)
    samples = weights @ per_chunk / per_chunk.size
    return {"mean": float(per_chunk.mean()), "by_seed": values.mean(axis=1).tolist(),
            "paired_chunk_bootstrap95_given_seeds": np.quantile(samples, (.025, .975)).tolist()}


def summarize(resources, factorial, calibration=None):
    base = resources[0]
    training = factorial["training"]
    by_fit = {}
    for resource in resources:
        assert resource["model"] == factorial["model"] and resource["revision"] == factorial["revision"]
        for attr in ("gpu", "token_manifest_sha256", "chunk_indices", "scored_tokens_per_chunk"):
            assert resource[attr] == base[attr]
        for ident, fit in resource["fits"].items():
            assert ident not in by_fit
            if ident == BASE:
                assert fit["adapter_sha256"] is None
            else:
                assert fit["adapter_sha256"] == training[ident]["adapter_sha256"]
                assert resource["token_manifest_sha256"] == training[ident]["token_manifest_sha256"]
            by_fit[ident] = fit
    recipes = ("reference", "high_lr_kl") if calibration else FACTORIAL_RECIPES
    expected = {key(seed, recipe, arm) for seed in SEEDS for recipe in recipes for arm in ARMS}
    assert by_fit.keys() == expected | {BASE}
    if calibration:
        assert calibration["model"] == base["model"]
        assert set(calibration["fits"]) == expected
        assert all(by_fit[ident]["scale"] == f["selected"]["scale"] for ident, f in calibration["fits"].items())
    results = {}
    for domain, ids in base["chunk_indices"].items():
        assert len(set(ids)) == len(ids) == 64
        weights = np.random.default_rng(42053).multinomial(64, [1/64] * 64, size=20000)
        untouched = np.asarray(by_fit[BASE]["domains"][domain]["nll_per_chunk"])
        results[domain] = {"untouched_nll": float(untouched.mean()),
                           "untouched_perplexity": float(np.exp(untouched.mean())), "arms": {}, "paired": {}}
        for recipe in recipes:
            losses = {}
            for arm in ARMS:
                metrics = [by_fit[key(seed, recipe, arm)]["domains"][domain] for seed in SEEDS]
                nll = np.asarray([m["nll_per_chunk"] for m in metrics])
                kl = np.asarray([m["kl_per_chunk"] for m in metrics])
                assert nll.shape == kl.shape == (len(SEEDS), 64)
                assert np.isfinite(nll).all() and np.isfinite(kl).all()
                assert nll.min() >= 0 and kl.min() >= -1e-5
                losses[arm] = nll
                results[domain]["arms"][f"{recipe}-{arm}"] = {
                    "nll": float(nll.mean()), "perplexity": float(np.exp(nll.mean())), "kl": float(kl.mean()),
                    "vs_untouched_nll": contrast(nll - untouched[None, :], weights),
                }
            results[domain]["paired"][recipe] = contrast(losses["kam"] - losses["dot"], weights)
    return {k: base[k] for k in ("model", "revision", "gpu", "token_manifest_sha256",
                                 "chunk_indices", "scored_tokens_per_chunk")} | {"domains": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resources", nargs=4, type=Path)
    parser.add_argument("--factorial", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    data = [json.loads(p.read_text()) for p in args.resources]
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    results = summarize(data, json.loads(args.factorial.read_text()), calibration)
    args.output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
