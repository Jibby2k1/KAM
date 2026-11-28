#!/usr/bin/env python3
"""Verify query-window provenance and replay paired quality and timing."""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from summarize_kam_block_shortlist import comparison, quality, rates, score_matrix

WINDOWS = ("tail-32", "tail-256", "before-tail-256")
CONTROLS = ("flash", "protected")
WIDTHS = ("512", "2048", "4096", "8192")


def timings(profile, modes):
    assert set(profile["full_prefill"]) == set(WIDTHS)
    scores = {}
    for width, arms in profile["full_prefill"].items():
        assert set(arms) == set(modes)
        scores[width] = {}
        for mode, row in arms.items():
            wall, gpu = [np.asarray(row[f"{k}_ms"]) for k in ("wall", "cuda")]
            assert len(wall) == len(gpu) == 15 and all(np.isfinite(x).all() and min(x) > 0 for x in (wall, gpu))
            assert row["incremental_allocated_bytes"] >= 0
            scores[width][mode] = {"p50_ms": float(np.median(wall)), "p95_ms": float(np.quantile(wall, .95)),
                "cuda_p50_ms": float(np.median(gpu)), "peak_bytes": row["incremental_allocated_bytes"]}
    return scores


def summarize(work, manifest_path, backbone, protected):
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    profile_path = work / "query-window-profile.json"
    profile = json.loads(profile_path.read_text())
    assert profile["manifest_sha256"] == manifest_hash and profile["seed"] == SEEDS[0]
    assert set(profile["parity"]) == set(WINDOWS) and max(profile["parity"].values()) < .012
    refs = {}
    for mode, path in zip(CONTROLS, (backbone, protected), strict=True):
        with gzip.open(path, "rt") as source:
            refs[mode] = json.load(source)
    book_ids = [r["id"] for r in manifest["pg19"]]
    labels = [(r["id"], r["family"], index) for r in manifest["retrieval"] for index in (0, 1)]
    families = [r["family"] for r in manifest["retrieval"]]
    assert len(book_ids) == 48 and len(families) == 320
    modes = (*CONTROLS, *WINDOWS)
    values = {key: [] for key in modes}
    hashes = {profile_path.stem: digest(profile_path)}
    for seed in SEEDS:
        baseline = json.loads(refs["flash"][f"backbone-{seed}-0"])
        anchor = json.loads(refs["protected"][f"first-{seed}-32"])
        check = baseline["checkpoint_sha256"]
        if seed == SEEDS[0]:
            assert profile["checkpoint_sha256"] == check
        inputs = dict(flash=baseline, protected=anchor)
        for mode in WINDOWS:
            path = work / f"window-{seed}-{mode}.json"
            row = json.loads(path.read_text())
            assert row["variant"] == mode
            inputs[mode] = row
            hashes[path.stem] = digest(path)
        for mode, row in inputs.items():
            assert (row["seed"], row["gpu"], row["manifest_sha256"], row["checkpoint_sha256"]) == (
                seed, profile["gpu"], manifest_hash, check)
            values[mode].append(score_matrix(row, book_ids, labels))
    values = {mode: quality(value) for mode, value in values.items()}
    score = {mode: rates(value) for mode, value in values.items()}
    effects = {mode: {ref: comparison(values[mode], values[ref], families) for ref in CONTROLS} for mode in WINDOWS}
    times = timings(profile, modes)
    # Freeze the best tail arm on this panel, holding a faster-than-Flash 4K cost constraint.
    eligible = [mode for mode in WINDOWS[:2] if times["4096"][mode]["p50_ms"] < times["4096"]["flash"]["p50_ms"]]
    choices = eligible or list(WINDOWS[:2])
    chosen = max(choices, key=lambda mode: (score[mode]["mirrored_accuracy"], -times["4096"][mode]["p50_ms"]))
    return {"manifest_sha256": manifest_hash, "backbone_sha256": digest(backbone), "protected_sha256": digest(protected),
            "raw_sha256": hashes, "gpu": profile["gpu"], "torch": profile["torch"], "parity": profile["parity"],
            "quality": score, "window_minus": effects, "tail_minus_earlier_equal_query_budget": comparison(
                values["tail-256"], values["before-tail-256"], families), "timings": times,
            "transfer_arm": chosen, "transfer_arm_cost_gate_passed": chosen in eligible}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "manifest", "backbone", "protected", "summary"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    write_json(a.summary, summarize(a.work, a.manifest, a.backbone, a.protected))
