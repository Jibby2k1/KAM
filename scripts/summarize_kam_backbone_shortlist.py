#!/usr/bin/env python3
"""Verify paired backbone depth runs and summarize quality and full-reader cost."""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from summarize_kam_block_shortlist import comparison, quality, rates, score_matrix
from summarize_kam_pretrained_endpoint_selection import family_difference

DEPTHS = (0, 8, 16, 32)
WIDTHS = (512, 2048, 4096, 8192)


def layout(depth):
    return list(range(32 // depth - 1, 32, 32 // depth)) if depth else []


def timings(profile):
    trials = profile["full_prefill"]
    assert set(trials) == set(map(str, WIDTHS))
    result = {}
    for width, rows in trials.items():
        assert set(rows) == set(map(str, DEPTHS))
        result[width] = {}
        for depth, row in rows.items():
            cuda, wall = [np.array(row[f"{clock}_ms"]) for clock in ("cuda", "wall")]
            assert len(cuda) == len(wall) == 15
            assert all(np.isfinite(x).all() and min(x) > 0 for x in (cuda, wall))
            assert row["incremental_allocated_bytes"] >= 0
            result[width][depth] = {"cuda_p50_ms": float(np.median(cuda)), "wall_p50_ms": float(np.median(wall)),
                "wall_p95_ms": float(np.quantile(wall, .95)), "peak_bytes": row["incremental_allocated_bytes"]}
        base = result[width]["0"]
        for cost in result[width].values():
            cost["wall_p50_change_vs_flash"] = cost["wall_p50_ms"] / base["wall_p50_ms"] - 1
            cost["peak_delta_vs_flash_bytes"] = cost["peak_bytes"] - base["peak_bytes"]
    return result


def summarize(work, manifest_path, previous_path, source_audit_path):
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    assert manifest["context"] == 4096 and manifest["scored_tokens_per_window"] == 1024
    book_ids = [r["id"] for r in manifest["pg19"]]
    families = [r["family"] for r in manifest["retrieval"]]
    labels = [(r["id"], r["family"], order) for r in manifest["retrieval"] for order in (0, 1)]
    assert len(book_ids) == 48 and len(families) == 320
    audit = json.loads(source_audit_path.read_text())
    overlaps = {item for source in audit.values() for item in source["matches"]["test"]}
    clean = np.array([not overlaps.intersection([*r["support_ids"], r["foil_id"], *r["filler_ids"]])
                      for r in manifest["retrieval"]])
    clean_families = [f for f, keep in zip(families, clean, strict=True) if keep]
    with gzip.open(previous_path, "rt") as source:
        archive = json.load(source)
    prior = {seed: json.loads(archive[f"eval-{seed}-top4-linear"]) for seed in SEEDS}
    profile_file = work / "backbone-profile.json"
    profile = json.loads(profile_file.read_text())
    wide_file = work / "wide-check.json"
    wide = json.loads(wide_file.read_text())
    assert (wide["width"], wide["heads"], wide["kv"], wide["route_checks"], wide["gpu"]) == (8192, 32, 8, 96, profile["gpu"])
    assert wide["kernel_sha256"] == digest(Path(__file__).with_name("kam_backbone_block_attention.py"))
    assert set(wide["max_abs_delta"]) == {"4096", "6144", "8160"}
    assert profile["manifest_sha256"] == manifest_hash and profile["seed"] == SEEDS[0]
    assert profile["checkpoint_sha256"] == prior[SEEDS[0]]["checkpoint_sha256"]
    assert (profile["block"], profile["group"], profile["query_tile"], profile["budget"]) == (64, 16, 32, 8)
    assert profile["depth_layers"] == {str(depth): layout(depth) for depth in DEPTHS}
    assert set(profile["checks"]) == {"103", "512", "777"}
    assert max(abs(x) for x in profile["model_check"]["all_layer_full_prefix_nll_change"]) < .015
    stats, previous = {}, []
    hashes = {p.stem: digest(p) for p in (profile_file, wide_file)}
    for seed in SEEDS:
        row = prior[seed]
        assert row["seed"] == seed and row["manifest_sha256"] == manifest_hash and row["gpu"] == profile["gpu"]
        assert row["scored_backend"] == "sparse" and row["audit"]["passed"]
        previous.append(score_matrix(row, book_ids, labels))
    for depth in DEPTHS:
        runs = []
        for seed in SEEDS:
            path = work / f"backbone-{seed}-{depth}.json"
            row = json.loads(path.read_text())
            assert (row["seed"], row["depth"], row["gpu"], row["manifest_sha256"]) == (seed, depth, profile["gpu"], manifest_hash)
            assert row["checkpoint_sha256"] == prior[seed]["checkpoint_sha256"] and row["layers"] == layout(depth)
            hashes[path.stem] = digest(path)
            runs.append(score_matrix(row, book_ids, labels))
        stats[depth] = quality(runs)
    base = stats[0]
    quality_results = {}
    per_layer_fraction = sum(min(7, i // 64) * 64 + i % 64 + 1 for i in range(4096)) / math.comb(4097, 2)
    for depth, row in stats.items():
        delta = comparison(row, base, families)
        quality_results[depth] = {"rates": rates(row), "delta_vs_flash": delta,
            "ppl_relative_change": math.expm1(delta["nll"]["mean_delta"]),
            "eligible_token_pair_fraction_4k": (32 - depth + depth * per_layer_fraction) / 32,
            "clean_mirrored_vs_flash": family_difference(row["mirrored"][:, clean], base["mirrored"][:, clean], clean_families)}
    return {"manifest_sha256": manifest_hash, "prior_results_sha256": digest(previous_path),
            "source_audit_sha256": digest(source_audit_path), "clean_questions": int(clean.sum()), "raw_sha256": hashes,
            "gpu": profile["gpu"], "torch": profile["torch"], "triton": profile["triton"], "transformers": profile["transformers"],
            "kernel_checks": profile["checks"], "wide_check": wide, "model_check": profile["model_check"],
            "depth_layers": profile["depth_layers"],
            "forced_flash_vs_prior_dense": comparison(base, quality(previous), families),
            "quality": quality_results, "full_prefill": timings(profile)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "manifest", "previous", "source_audit", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    write_json(a.output, summarize(a.work, a.manifest, a.previous, a.source_audit))
