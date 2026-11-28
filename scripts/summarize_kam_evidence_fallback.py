#!/usr/bin/env python3
"""Verify development-only selection and replay the distant-evidence comparison."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_evidence_fallback import HEADS, LAYERS, N_FLASH, QUERIES, REGIONS, identities, read
from kam_math_kl_protocol import write_json
from summarize_kam_block_shortlist import comparison, quality, rates, score_matrix

MODES = {"flash", "periodic-16", "selected-16", "protected-32"}
WIDTHS = ("512", "2048", "4096", "8192")


def stats(pair):
    full, kept = map(float, pair)
    return {"dense_mass": full, "retained_mass": kept, "omitted_mass": full-kept,
            "retained_fraction": kept/full if full else None}


def report_mass(mass, sparse_layers):
    policies = {"protected-32": mass[..., 1], "flash": mass[..., 0]}
    for name, indices in sparse_layers.items():
        protected = np.array([i in indices for i in range(LAYERS)])[:, None, None, None]
        policies[name] = np.where(protected, mass[..., 1], mass[..., 0])
    global_rows, layer_rows = {}, {}
    for qi, query in enumerate(QUERIES):
        for ri, region in enumerate(REGIONS):
            full = mass[:, :, qi, ri, 0]
            key = f"{query}/{region}"
            global_rows[key] = {mode: stats((full.mean(), m[:, :, qi, ri].mean())) for mode, m in policies.items()}
            layer_rows[key] = [{mode: stats((full[i].mean(), m[i, :, qi, ri].mean())) for mode, m in policies.items()}
                               for i in range(LAYERS)]
    return {"pooled": global_rows, "by_layer": layer_rows}


def summarize(work, spans_path, selection_path, manifest_path, backbone_path, protected_path, output_csv):
    spans, selection, base_manifest = map(read, (spans_path, selection_path, manifest_path))
    manifest_hash, span_hash, policy_hash = map(digest, (manifest_path, spans_path, selection_path))
    assert spans["holdout_manifest_sha256"] == manifest_hash and selection["spans_sha256"] == span_hash
    assert len(spans["dev"]) == 96 and len(spans["holdout"]) == 640
    assert not ({r["family"] for r in spans["dev"]} & {r["family"] for r in spans["holdout"]})
    assert not ({r["id"] for r in spans["dev"]} & {r["id"] for r in spans["holdout"]})
    book_ids = [b["id"] for b in base_manifest["pg19"]]
    families = [r["family"] for r in base_manifest["retrieval"]]
    labels = [(r["id"], r["family"], slot) for r in base_manifest["retrieval"] for slot in (0, 1)]
    assert identities(spans["holdout"]) == [list(row) for row in labels]
    with gzip.open(backbone_path, "rt") as file:
        baseline = json.load(file)
    with gzip.open(protected_path, "rt") as file:
        older = json.load(file)
    profile_path = work / "fallback-profile.json"
    profile = read(profile_path)
    gpu = profile["gpu"]
    hashes = {profile_path.stem: digest(profile_path)}
    mass = {split: [] for split in ("dev", "holdout")}
    panels = {m: [] for m in MODES}
    shortlist = selection["shortlist_layers_zero_based"]
    assert len(shortlist) == N_FLASH and len(selection["flash_layers_zero_based"]) == N_FLASH
    assert set(shortlist) == set(range(LAYERS)) - set(selection["flash_layers_zero_based"])
    sparse_layers = {"periodic-16": list(range(1, LAYERS, 2)), "selected-16": shortlist}
    assert profile["modes"] == {"flash": [], "protected-32": list(range(LAYERS)), **sparse_layers}
    assert (profile["seed"], profile["manifest_sha256"], profile["spans_sha256"], profile["selection_sha256"]) == (
        SEEDS[0], manifest_hash, span_hash, policy_hash)
    for seed in SEEDS:
        raw = json.loads(baseline[f"backbone-{seed}-0"])
        checkpoint = raw["checkpoint_sha256"]
        if seed == SEEDS[0]:
            assert checkpoint == profile["checkpoint_sha256"]
        for split in mass:
            path = work / f"{split}-mass-{seed}.json"
            row = read(path)
            hashes[path.stem] = digest(path)
            assert (row["seed"], row["gpu"], row["checkpoint_sha256"], row["manifest_sha256"], row["spans_sha256"]) == (
                seed, gpu, checkpoint, manifest_hash, span_hash)
            assert row["presentations"] == identities(spans[split])
            assert tuple(row["queries"]) == QUERIES and tuple(row["regions"]) == REGIONS
            assert row["selection_sha256"] == (None if split == "dev" else policy_hash)
            values = np.array(row["sum_mass"])
            assert values.shape == (LAYERS, HEADS, len(QUERIES), len(REGIONS), 2)
            assert np.isfinite(values).all() and (values >= 0).all() and (values[..., 0] + .001 >= values[..., 1]).all()
            mass[split].append(values / len(spans[split]))
            if split == "dev":
                assert hashes[path.stem] == selection["dev_mass_sha256"][path.stem]
        result_path = work / f"fallback-{seed}.json"
        result = read(result_path)
        hashes[result_path.stem] = digest(result_path)
        assert (result["selection_sha256"], result["spans_sha256"], result["shortlist_layers"]) == (policy_hash, span_hash, shortlist)
        saved = {"flash": raw, "periodic-16": json.loads(older[f"first-{seed}-16"]),
                 "protected-32": json.loads(older[f"first-{seed}-32"]), "selected-16": result}
        for m, row in saved.items():
            assert (row["seed"], row["gpu"], row["checkpoint_sha256"], row["manifest_sha256"]) == (
                seed, gpu, checkpoint, manifest_hash)
            panels[m].append(score_matrix(row, book_ids, labels))
    mass = {split: np.mean(data, axis=0) for split, data in mass.items()}
    i = REGIONS.index("earlier_outside_first")
    merit = (mass["dev"][..., i, 0] - mass["dev"][..., i, 1]).mean(axis=(1, 2))
    np.testing.assert_allclose(merit, selection["mean_omitted_mass_by_layer"], atol=1e-14, rtol=0)
    ranking = sorted(range(LAYERS), key=lambda j: (-merit[j], j))
    assert selection["ranked_layers_zero_based"] == ranking
    assert selection["flash_layers_zero_based"] == sorted(ranking[:N_FLASH])

    with output_csv.open("w", newline="") as stream:
        csv_out = csv.writer(stream, lineterminator="\n")
        csv_out.writerow(("panel", "layer", "head", "query", "span", "dense_mass", "retained_mass", "retained_fraction"))
        for split, out in mass.items():
            for layer, head, qi, ri in np.ndindex(out.shape[:-1]):
                pair = stats(out[layer, head, qi, ri])
                csv_out.writerow((split, layer+1, head+1, QUERIES[qi], REGIONS[ri], pair["dense_mass"],
                                  pair["retained_mass"], pair["retained_fraction"]))

    panels = {mode: quality(rows) for mode, rows in panels.items()}
    quality_result = {mode: rates(panel) for mode, panel in panels.items()}
    comparisons = {mode: comparison(panels["selected-16"], panel, families) for mode, panel in panels.items() if mode != "selected-16"}
    trials = profile["full_prefill"]
    assert set(trials) == set(WIDTHS)
    timings = {}
    for width, trial in trials.items():
        assert set(trial) == MODES
        timings[width] = {}
        for mode, r in trial.items():
            wall, cuda = np.array(r["wall_ms"]), np.array(r["cuda_ms"])
            assert len(wall) == len(cuda) == 15 and np.isfinite(wall).all() and (wall > 0).all()
            assert np.isfinite(cuda).all() and (cuda > 0).all() and r["incremental_allocated_bytes"] >= 0
            timings[width][mode] = {"wall_p50_ms": float(np.median(wall)), "wall_p95_ms": float(np.quantile(wall, .95)),
                                    "cuda_p50_ms": float(np.median(cuda)), "peak_bytes": r["incremental_allocated_bytes"]}
    return {"manifest_sha256": manifest_hash, "spans_sha256": span_hash, "selection_sha256": policy_hash,
            "baseline_archive_sha256": digest(backbone_path), "protected_archive_sha256": digest(protected_path),
            "raw_sha256": hashes, "layer_head_sha256": digest(output_csv), "gpu": gpu, "torch": profile["torch"],
            "flash_layers": [r+1 for r in selection["flash_layers_zero_based"]],
            "quality": quality_result, "selected_minus": comparisons, "ppl_change_vs_flash": math.expm1(comparisons["flash"]["nll"]["mean_delta"]),
            "mass": {split: report_mass(v, sparse_layers) for split, v in mass.items()}, "timings": timings}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "spans", "selection", "manifest", "backbone", "protected", "summary", "layer_head_csv"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    write_json(a.summary, summarize(a.work, a.spans, a.selection, a.manifest, a.backbone, a.protected, a.layer_head_csv))
