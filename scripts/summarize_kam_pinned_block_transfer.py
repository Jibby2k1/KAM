#!/usr/bin/env python3
"""Replay fixed prefix protection at matched shortlist and context budgets."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from prepare_kam_attention_transfer import PROTOCOL, QUESTIONS, WIDTHS, read
from summarize_kam_block_shortlist import comparison, quality, rates, score_matrix

PINS = {"prefix-2": [0, 1], "prefix-4": [0, 1, 2, 3], "late-4": [0, 8, 9, 10]}
BASE = {"flash": "flash", "protected": "protected", "tail-256": "selected"}


def summarize(work, manifest_path, prior_path, audit_path):
    fixture, audit = map(read, (manifest_path, audit_path))
    identity = digest(manifest_path)
    assert fixture["protocol"] == PROTOCOL and audit["manifest_sha256"] == identity
    assert audit["tokenizer_sha256"] == fixture["tokenizer_sha256"] and audit["presentations"] == QUESTIONS*2
    profile_path = work / "pinned-profile.json"
    profile = read(profile_path)
    assert profile["pins"] == PINS and profile["variant"] == "benchmark"
    assert profile["manifest_sha256"] == identity and profile["seed"] == SEEDS[0]
    gpu = profile["gpu"]
    with gzip.open(prior_path, "rt") as stream:
        original = json.load(stream)
    base = {name: [json.loads(original[f"transfer-{s}-{arm}"]) for s in SEEDS] for name, arm in BASE.items()}
    checks = {row["seed"]: row["checkpoint_sha256"] for row in base["flash"]}
    assert profile["checkpoint_sha256"] == checks[SEEDS[0]]
    hashes, outcomes = {profile_path.stem: digest(profile_path)}, {}
    for name, rows in base.items():
        for s, row in zip(SEEDS, rows, strict=True):
            assert (row["seed"], row["gpu"], row["checkpoint_sha256"], row["transfer_manifest_sha256"], row["arm"]) == (
                s, gpu, checks[s], identity, BASE[name])
        outcomes[name] = rows
    for arm in PINS:
        outcomes[arm] = []
        for seed in SEEDS:
            path = work / f"pinned-{seed}-{arm}.json"
            row = read(path)
            hashes[path.stem] = digest(path)
            assert (row["seed"], row["variant"], row["gpu"], row["checkpoint_sha256"], row["manifest_sha256"], row["pins"]) == (
                seed, arm, gpu, checks[seed], identity, PINS)
            outcomes[arm].append(row)

    by_width = {}
    books = [r["id"] for r in fixture["pg19"]]
    for width in map(str, WIDTHS):
        questions = fixture["fixture"][width]["questions"]
        labels = [(row["id"], row["family"], o) for row in questions for o in (0, 1)]
        families = [r["family"] for r in questions]
        assert len(questions) == QUESTIONS
        records = {arm: quality([score_matrix(r["outcomes"][width], books, labels, windows=1) for r in panels])
                   for arm, panels in outcomes.items()}
        timing = {}
        for name, row in profile["outcomes"][width].items():
            wall, cuda = map(np.array, (row["wall_ms"], row["cuda_ms"]))
            assert len(wall) == len(cuda) == 15 and all(np.isfinite(a).all() and min(a)>0 for a in (wall, cuda))
            timing[name] = {"wall_p50_ms": float(np.median(wall)), "wall_p95_ms": float(np.quantile(wall, .95)),
                            "cuda_p50_ms": float(np.median(cuda)), "peak_bytes": row["incremental_allocated_bytes"]}
        by_width[width] = {"quality": {arm: rates(row) for arm, row in records.items()}, "timing": timing,
                          "paired": {f"{a}_minus_{b}": comparison(records[a], records[b], families)
                             for a, b in (("prefix-2", "protected"), ("prefix-4", "protected"),
                                          ("prefix-4", "late-4"), ("late-4", "protected"))}}
    return {"manifest_sha256": identity, "prior_archive_sha256": digest(prior_path), "raw_sha256": hashes,
            "audit_sha256": digest(audit_path), "pins": PINS, "gpu": gpu, "coverage": audit, "widths": by_width}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "manifest", "prior", "audit", "output"):
        cli.add_argument(name, type=Path)
    a = cli.parse_args()
    write_json(a.output, summarize(a.work, a.manifest, a.prior, a.audit))
