#!/usr/bin/env python3
"""Replay layerwise retained attention mass and the protected-first quality runs."""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from summarize_kam_backbone_shortlist import layout
from summarize_kam_block_shortlist import comparison, quality, rates, score_matrix
from summarize_kam_pretrained_endpoint_selection import family_difference

DEPTHS = (8, 16, 32)
WIDTHS = (512, 2048, 4096, 8192)
BINS = {"all": 65, "pruned": 57, "last_quarter": 17, "final_query": 1}
FIELDS = ("rows", "ordinary_mass_sum", "protected_mass_sum", "first_block_mass_sum", "omitted_rows",
          "omitted_first_mass_sum", "displaced_mass_sum", "ordinary_below_half", "protected_below_half")


def measure_mass(counts):
    n, ordinary, kept, first, missing, omitted_mass, displaced, low, low_kept = counts
    return {"head_queries": int(n), "ordinary": ordinary/n, "protected": kept/n, "gain": (kept-ordinary)/n,
            "first_mass": first/n, "first_missing_rate": missing/n, "first_missing_head_queries": int(missing),
            "first_mass_when_missing": omitted_mass/missing if missing else None,
            "displaced_mass_when_missing": displaced/missing if missing else None,
            "ordinary_below_half": low/n, "protected_below_half": low_kept/n}


def read_mass(work, manifest, expected_hash, checkpoints, gpu, hashes):
    samples = ([("pg19", row["id"], None) for row in manifest["pg19"][:8]]
               + [("musique", row["id"], order) for row in manifest["retrieval"][:8] for order in (0, 1)])
    outcome = {label: np.zeros((32, len(BINS), len(FIELDS))) for label in ("combined", "pg19", "musique")}
    expected_counts = np.array(list(BINS.values())) * 32
    for seed in SEEDS:
        file = work / f"mass-{seed}.json"
        raw = json.loads(file.read_text())
        hashes[file.stem] = digest(file)
        assert (raw["seed"], raw["gpu"], raw["manifest_sha256"], raw["checkpoint_sha256"]) == (
            seed, gpu, expected_hash, checkpoints[seed])
        assert tuple(raw["fields"]) == FIELDS and raw["rows_per_prompt"] == 65
        assert [(o["domain"], o["id"], o["order"]) for o in raw["observations"]] == samples
        for row in raw["observations"]:
            assert all(set(r) == set(BINS) for r in row["mass"])
            data = np.array([[r[b] for b in BINS] for r in row["mass"]])
            assert data.shape == (32, len(BINS), len(FIELDS)) and np.isfinite(data).all()
            assert (data[:, :, 0] == expected_counts).all()
            assert (data >= 0).all() and (data <= data[:, :, :1] + .002).all()
            assert (data[:, :, 5:7] <= data[:, :, 4:5] + .002).all()
            np.testing.assert_allclose(data[:, :, 2]-data[:, :, 1], data[:, :, 5]-data[:, :, 6], rtol=0, atol=.002)
            outcome["combined"] += data
            outcome[row["domain"]] += data
    return {domain: {"all_layers": {name: measure_mass(values[:, i].sum(axis=0)) for i, name in enumerate(BINS)},
                     "by_layer": {str(layer+1): {name: measure_mass(row[i]) for i, name in enumerate(BINS)}
                                  for layer, row in enumerate(values)}}
            for domain, values in outcome.items()}


def read_timings(profile):
    modes = {"flash", *(f"{mode}-{depth}" for mode in ("ordinary", "first") for depth in DEPTHS)}
    assert set(profile["full_prefill"]) == set(map(str, WIDTHS))
    times = {}
    for width, trial in profile["full_prefill"].items():
        assert set(trial) == modes
        series = {}
        for mode, r in trial.items():
            cuda, wall = np.array(r["cuda_ms"]), np.array(r["wall_ms"])
            assert len(cuda) == len(wall) == 15 and all(np.isfinite(x).all() and min(x) > 0 for x in (cuda, wall))
            assert r["incremental_allocated_bytes"] >= 0
            series[mode] = {"wall_p50_ms": float(np.median(wall)), "wall_p95_ms": float(np.quantile(wall, .95)),
                            "cuda_p50_ms": float(np.median(cuda)), "peak_bytes": r["incremental_allocated_bytes"]}
        for info in series.values():
            info["wall_p50_change_vs_flash"] = info["wall_p50_ms"] / series["flash"]["wall_p50_ms"] - 1
        times[width] = series
    return times


def summarize(work, manifest_path, baseline_path, audit_path):
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    assert manifest["context"] == 4096 and manifest["scored_tokens_per_window"] == 1024
    book_ids = [r["id"] for r in manifest["pg19"]]
    families = [r["family"] for r in manifest["retrieval"]]
    labels = [(r["id"], r["family"], expected) for r in manifest["retrieval"] for expected in (0, 1)]
    assert len(book_ids) == 48 and len(families) == 320
    audit = json.loads(audit_path.read_text())
    overlaps = {entry for source in audit.values() for entry in source["matches"]["test"]}
    clean = np.array([not overlaps.intersection([*r["support_ids"], r["foil_id"], *r["filler_ids"]])
                      for r in manifest["retrieval"]])
    clean_families = [family for family, keep in zip(families, clean, strict=True) if keep]
    with gzip.open(baseline_path, "rt") as source:
        old = json.load(source)
    profile_path = work / "first-profile.json"
    profile = json.loads(profile_path.read_text())
    gpu = profile["gpu"]
    assert profile["manifest_sha256"] == manifest_hash and profile["seed"] == SEEDS[0]
    assert profile["checks"]["512"]["intervened_head_rows"] == 0 and profile["checks"]["1024"]["intervened_head_rows"] > 0
    assert max(map(abs, profile["model_check"]["all_layer_full_prefix_nll_change"])) < .015
    hashes = {profile_path.stem: digest(profile_path)}
    checkpoints = {}
    ordinary = {}
    for depth in (0, *DEPTHS):
        panels = []
        for seed in SEEDS:
            row = json.loads(old[f"backbone-{seed}-{depth}"])
            assert (row["seed"], row["depth"], row["gpu"], row["manifest_sha256"], row["layers"]) == (
                seed, depth, gpu, manifest_hash, layout(depth))
            checkpoints.setdefault(seed, row["checkpoint_sha256"])
            assert row["checkpoint_sha256"] == checkpoints[seed]
            panels.append(score_matrix(row, book_ids, labels))
        ordinary[depth] = quality(panels)
    assert profile["checkpoint_sha256"] == checkpoints[SEEDS[0]]
    effects = {"flash": rates(ordinary[0]), "depth": {}}
    for depth in DEPTHS:
        panels = []
        for seed in SEEDS:
            p = work / f"first-{seed}-{depth}.json"
            row = json.loads(p.read_text())
            assert (row["seed"], row["depth"], row["gpu"], row["manifest_sha256"], row["layers"]) == (
                seed, depth, gpu, manifest_hash, layout(depth))
            assert row["checkpoint_sha256"] == checkpoints[seed]
            hashes[p.stem] = digest(p)
            panels.append(score_matrix(row, book_ids, labels))
        protected = quality(panels)
        flash_gap = comparison(protected, ordinary[0], families)
        protect_gain = comparison(protected, ordinary[depth], families)
        old_excess = (ordinary[depth]["nll"] - ordinary[0]["nll"]).mean()
        effects["depth"][depth] = {"ordinary": rates(ordinary[depth]), "protected": rates(protected),
            "protected_vs_flash": flash_gap, "protected_vs_ordinary": protect_gain,
            "ppl_change_vs_flash": math.expm1(flash_gap["nll"]["mean_delta"]),
            "fraction_of_excess_nll_removed": float(-protect_gain["nll"]["mean_delta"]/old_excess),
            "clean_mirrored_vs_flash": family_difference(protected["mirrored"][:, clean],
                ordinary[0]["mirrored"][:, clean], clean_families)}

    mass = read_mass(work, manifest, manifest_hash, checkpoints, gpu, hashes)
    return {"manifest_sha256": manifest_hash, "baseline_archive_sha256": digest(baseline_path),
            "source_audit_sha256": digest(audit_path), "gpu": gpu, "torch": profile["torch"], "triton": profile["triton"],
            "clean_questions": int(clean.sum()), "raw_sha256": hashes, "kernel_checks": profile["checks"],
            "model_check": profile["model_check"], "mass": mass, "quality": effects, "full_prefill": read_timings(profile)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "manifest", "baseline", "source_audit", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    write_json(a.output, summarize(a.work, a.manifest, a.baseline, a.source_audit))
