#!/usr/bin/env python3
"""Validate and summarize held-out comparisons at the same raw-text KL cap."""
import argparse
import json
from pathlib import Path
from statistics import mean

from analyze_kam_adapter_strength import outcomes
from kam_adapter_preservation import ARMS, SEEDS, key

RECIPES = ("reference", "high_lr_kl")


def summarize(raw, factorial):
    training = factorial["training"]
    reference = raw[0]
    by_fit = {}
    for fit in raw:
        ident = key(fit["seed"], fit["recipe"], fit["arm"])
        assert ident not in by_fit and fit["adapter_sha256"] == training[ident]["adapter_sha256"]
        assert (fit["model"], fit["revision"]) == (factorial["model"], factorial["revision"])
        for attr in ("kl_cap", "gpu", "panels", "untouched"):
            assert fit[attr] == reference[attr]
        panel_stats = {}
        for name, pane in fit["panels"].items():
            assert pane["source_sha256"] == factorial["tasks"][name]["source_sha256"]
            panel_stats[name] = outcomes(fit["choices"][name], fit["untouched"][name], pane["gold"])
        eligible = [entry for entry in fit["sampled_validation"] if entry["forward_kl"] <= fit["kl_cap"]]
        best = max(eligible, key=lambda entry: entry["scale"])
        assert best["scale"] == fit["scale"]
        by_fit[ident] = {"selected": best, "panel_stats": panel_stats}
    assert by_fit.keys() == {key(seed, recipe, arm) for seed in SEEDS for recipe in RECIPES for arm in ARMS}
    means = {}
    for recipe in RECIPES:
        for arm in ARMS:
            fits = [by_fit[key(seed, recipe, arm)] for seed in SEEDS]
            means[f"{recipe}-{arm}"] = {
                "scale": mean(f["selected"]["scale"] for f in fits),
                "forward_kl": mean(f["selected"]["forward_kl"] for f in fits),
                "nll": {k: mean(f["selected"]["nll"][k] for f in fits)
                        for k in fits[0]["selected"]["nll"]},
                "panel_stats": {task: {k: mean(f["panel_stats"][task][k] for f in fits)
                                       for k in row}
                                for task, row in fits[0]["panel_stats"].items()},
            }
    return {"model": reference["model"], "revision": reference["revision"], "gpu": reference["gpu"],
            "kl_cap": reference["kl_cap"], "fits": by_fit, "seed_means": means}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", nargs=12, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(p.read_text()) for p in args.probes]
    result = summarize(rows, json.loads(args.summary.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
