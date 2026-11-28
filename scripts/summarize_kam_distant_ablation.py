#!/usr/bin/env python3
"""Replay signed mirrored margins and controlled earlier-evidence perturbations."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_distant_evidence_ablation import ARMS, EDITS, read
from kam_math_kl_protocol import write_json
from summarize_kam_pretrained_endpoint_selection import family_difference


def margin_panel(rows, expected, *, exact):
    lookup = {(r["id"], r["family"], r["expected"]): r for r in rows}
    if exact:
        assert list(lookup) == expected
    assert len(lookup) == len(rows)
    values = np.array([lookup[k]["margin"] for k in expected]).reshape(-1, 2)
    assert np.isfinite(values).all()
    return values[:, 1]-values[:, 0]


def continuous_interval(delta, family):
    groups, index = np.unique(family, return_inverse=True)
    count = np.bincount(index)
    values = np.bincount(index, weights=delta.mean(axis=0))
    draw = np.random.default_rng(821590).multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=20000)
    bootstrap = (draw @ values) / (draw @ count)
    return {"mean_delta": float(delta.mean()), "by_seed": delta.mean(axis=1).tolist(),
            "paired_family_bootstrap95_given_seeds": np.quantile(bootstrap, (.025, .975)).tolist()}


def summarize(work, fixture_path, prior_paths):
    fixture = read(fixture_path)
    fixture_hash = digest(fixture_path)
    archives = {}
    for arm, source in zip(ARMS, prior_paths, strict=True):
        with gzip.open(source, "rt") as file:
            archives[arm] = json.load(file)
    labels = [(r["id"], r["family"], order) for r in fixture["questions"] for order in (0, 1)]
    families = [r["family"] for r in fixture["questions"]]
    assert len(families) >= 240 and len(labels) == len(set(labels))
    control_key = {"flash": "backbone-{}-0", "protected": "first-{}-32", "tail-256": "window-{}-tail-256"}
    hashes, quality, raw = {}, {}, {}
    for arm in ARMS:
        values = {label: [] for label in ("original", *EDITS)}
        for seed in SEEDS:
            p = work / f"ablation-{seed}-{arm}.json"
            result = read(p)
            hashes[p.stem] = digest(p)
            baseline = json.loads(archives[arm][control_key[arm].format(seed)])
            assert (result["seed"], result["arm"], result["fixture_sha256"], result["gpu"], result["checkpoint_sha256"]) == (
                seed, arm, fixture_hash, baseline["gpu"], baseline["checkpoint_sha256"])
            assert baseline["manifest_sha256"] == fixture["holdout_manifest_sha256"]
            rows = {"original": baseline["retrieval"], **result["outcomes"]}
            for name, panel in rows.items():
                values[name].append(margin_panel(panel, labels, exact=name != "original"))
        by_variant = {name: np.stack(s) for name, s in values.items()}
        raw[arm] = by_variant
        control, earlier, original = [by_variant[name] for name in ("background", "earlier", "original")]
        quality[arm] = {"variants": {name: {"mirrored_accuracy": float((s > 0).mean()), "signed_margin": float(s.mean()),
                                              "mirrored_by_seed": (s > 0).mean(axis=1).tolist()}
                                     for name, s in by_variant.items()},
            "earlier_minus_background_margin": continuous_interval(earlier-control, families),
            "background_minus_original_margin": continuous_interval(control-original, families),
            "earlier_minus_background_correct": family_difference(earlier > 0, control > 0, families)}
    interventions = {arm: raw[arm]["background"] - raw[arm]["earlier"] for arm in ARMS}
    changes = {arm: (raw[arm]["background"] > 0).astype(np.int8) - (raw[arm]["earlier"] > 0).astype(np.int8)
               for arm in ARMS}
    comparisons = {baseline: {"margin": continuous_interval(interventions["tail-256"]-interventions[baseline], families),
        "mirrored": family_difference(changes["tail-256"], changes[baseline], families)} for baseline in ("flash", "protected")}
    return {"fixture_sha256": fixture_hash, "prior_archive_sha256": {a: digest(p) for a, p in zip(ARMS, prior_paths, strict=True)},
            "n": len(families), "families": len(set(families)), "raw_sha256": hashes, "quality": quality,
            "tail_minus_reference_evidence_dependence": comparisons}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("fixture", type=Path)
    p.add_argument("prior_paths", type=Path, nargs=3)
    p.add_argument("summary", type=Path)
    a = p.parse_args()
    write_json(a.summary, summarize(a.work, a.fixture, a.prior_paths))
