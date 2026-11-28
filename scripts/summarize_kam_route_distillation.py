#!/usr/bin/env python3
"""Replay held-out route coverage and the exact fixed-budget ceiling."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from prepare_kam_route_distillation import PROTOCOL, read
from summarize_kam_live_token_factorial import estimate

BASE = "protected"
MODES = ("fixed", "adjustable_mass", BASE, "moments", "moments-position", "minmax-bound", "oracle")


def summarize(work, manifest_path, backbone):
    manifest = read(manifest_path)
    assert manifest["protocol"] == PROTOCOL and manifest["context"] == 4096
    manifest_hash = digest(manifest_path)
    with gzip.open(backbone, "rt") as stream:
        prior = json.load(stream)
    fit, dev = ([r["id"] for r in manifest["groups"][k]] for k in ("fit", "dev"))
    assert len(set(fit + dev)) == len(fit + dev) and len(fit) + len(dev) + len(manifest["skipped"]) == 50
    arrays = {name: [] for name in MODES}
    hashes = {}
    for seed in SEEDS:
        path = work / f"route-fit-{seed}.json"
        row = read(path)
        old = json.loads(prior[f"backbone-{seed}-0"])
        hashes[path.stem] = digest(path)
        assert (row["seed"], row["gpu"], row["checkpoint_sha256"], row["input_manifest_sha256"]) == (
            seed, old["gpu"], old["checkpoint_sha256"], manifest_hash)
        assert row["fit_books"] == fit and [r["id"] for r in row["dev"]] == dev
        assert [r["windows"] for r in row["dev"]] == [len(r["blocks"]) for r in manifest["groups"]["dev"]]
        assert row["query_stride"] == 128 and row["flex_slots"] == 5 and row["ridge"] == .005
        for name in MODES:
            a = np.array([r["mean_mass"][name] for r in row["dev"]])
            assert a.shape == (len(dev), 32, 32) and np.isfinite(a).all()
            assert a.min() >= 0 and a.max() < 1 + 1e-5
            arrays[name].append(a)

    arrays = {name: np.stack(rs) for name, rs in arrays.items()}
    np.testing.assert_allclose(arrays["fixed"] + arrays["adjustable_mass"], 1, atol=2e-6)
    base = arrays[BASE]
    results = {}
    for name in MODES:
        a = arrays[name]
        if name != "adjustable_mass":
            assert np.min(arrays["oracle"] - a) > -2e-6
        delta = a - base
        results[name] = {"mean_retained_mass": float(a.mean()), "mean_by_seed": a.mean((1, 2, 3)).tolist(),
                         "mass_delta_vs_protected": estimate(delta.mean((2, 3))),
                         "mean_layer_head_delta": delta.mean((0, 1)).tolist()}
    learned = ("moments", "moments-position")
    best = max(learned, key=lambda name: results[name]["mean_retained_mass"])
    return {"manifest_sha256": manifest_hash, "backbone_archive_sha256": digest(backbone),
            "gpu": old["gpu"], "raw_sha256": hashes, "fit_books": len(fit), "development_books": len(dev),
            "largest_learned_mean_mass": best, "development": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "manifest", "backbone", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    write_json(a.output, summarize(a.work, a.manifest, a.backbone))
