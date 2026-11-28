#!/usr/bin/env python3
"""Replay paired shortlist scores, exact-key recovery, and measured GPU costs."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from summarize_kam_live_token_factorial import estimate
from summarize_kam_long_sparse import read_panel
from summarize_kam_pretrained_endpoint_selection import family_difference

BUDGETS = (4, 8, 16)
LENGTHS = (512, 2048, 4096, 8192)
REGIONS = ("first_quarter", "middle_half", "last_quarter")
DOMAINS = {"pg19": 4, "musique": 8}


def summarize_timings(payload, shapes, repetitions):
    assert set(payload) == set(shapes)
    trials = {}
    for width, rows in payload.items():
        assert set(rows) == {"flash", "exact", *map(str, BUDGETS)}
        results = {}
        for mode, row in rows.items():
            cuda = np.array(row["cuda_ms"])
            wall = np.array(row["wall_ms"])
            assert len(cuda) == len(wall) == repetitions
            assert all(np.isfinite(x).all() and min(x) > 0 for x in (cuda, wall))
            assert row["incremental_allocated_bytes"] >= 0
            results[mode] = {"cuda_p50_ms": float(np.median(cuda)), "wall_p50_ms": float(np.median(wall)),
                             "wall_p95_ms": float(np.quantile(wall, .95)), "peak_bytes": row["incremental_allocated_bytes"]}
        trials[width] = results
    return trials


def recovery_table(counts):
    hits, keys, complete, rows = map(int, counts)
    assert 0 <= complete <= rows and 0 <= hits <= keys and keys == 4 * rows
    return {"keys_recovered": hits, "exact_keys": keys, "complete_sets": complete, "head_rows": rows,
            "key_recall": hits / keys, "all_four": complete / rows}


def score_matrix(panel, book_ids, labels):
    losses, margins, consistent = read_panel(panel, book_ids, labels)
    margin = margins.reshape(-1, 2)
    return losses.mean(axis=1), margin[:, 1] > margin[:, 0], consistent


def quality(matrix):
    nll, mirrored, both = (np.stack(values) for values in zip(*matrix, strict=True))
    return {"nll": nll, "mirrored": mirrored, "both": both}


def rates(a):
    return {"perplexity": math.exp(a["nll"].mean()), "nll_by_seed": a["nll"].mean(axis=1).tolist(),
            "mirrored_accuracy": float(a["mirrored"].mean()), "mirrored_by_seed": a["mirrored"].mean(axis=1).tolist(),
            "both_orders_accuracy": float(a["both"].mean())}


def comparison(mode, other, families):
    return {"nll": estimate(mode["nll"] - other["nll"]),
            "mirrored": family_difference(mode["mirrored"], other["mirrored"], families)}


def summarize(work, manifest_path, previous_path, audit_path):
    manifest = json.loads(manifest_path.read_text())
    with gzip.open(previous_path, "rt") as file:
        previous = json.load(file)
    audit = json.loads(audit_path.read_text())
    previous_hashes = {}
    profile_path = work / "block-profile.json"
    profile = json.loads(profile_path.read_text())
    manifest_hash = digest(manifest_path)
    assert profile["manifest_sha256"] == manifest_hash
    assert (profile["block"], profile["proxies"], profile["primary_budget"]) == (64, 4, 8)
    assert profile["seed"] == SEEDS[0]
    assert {tuple(map(int, shape.split("-"))) for shape in profile["checks"]} == {
        (1, 1, 1), (2, 97, 173), (1, 257, 257), (1, 1, 512), (1, 512, 512)}
    book_ids = [r["id"] for r in manifest["pg19"]]
    labels = [(r["id"], r["family"], index) for r in manifest["retrieval"] for index in (0, 1)]
    families = [r["family"] for r in manifest["retrieval"]]
    assert len(book_ids) == 48 and len(families) == 320
    assert manifest["context"] == 4096 and manifest["scored_tokens_per_window"] == 1024
    overlaps = {x for source in audit.values() for x in source["matches"]["test"]}
    clean = np.array([not overlaps.intersection([*r["support_ids"], r["foil_id"], *r["filler_ids"]])
                       for r in manifest["retrieval"]])

    reference, soft, checkpoints = [], [], {}
    for seed in SEEDS:
        for arm, series in (("top4-linear", reference), ("soft-linear", soft)):
            name = f"eval-{seed}-{arm}"
            previous_hashes[name] = hashlib.sha256(previous[name].encode()).hexdigest()
            panel = json.loads(previous[name])
            assert (panel["seed"], panel["arm"], panel["gpu"], panel["manifest_sha256"]) == (seed, arm, profile["gpu"], manifest_hash)
            if arm == "top4-linear":
                assert panel["scored_backend"] == "sparse" and panel["audit"]["passed"]
                checkpoints[seed] = panel["checkpoint_sha256"]
            series.append(score_matrix(panel, book_ids, labels))
    assert profile["checkpoint_sha256"] == checkpoints[SEEDS[0]]
    reference, soft = quality(reference), quality(soft)
    clean_families = [family for family, keep in zip(families, clean, strict=True) if keep]
    hashes = {profile_path.stem: digest(profile_path)}
    results = {}
    for budget in BUDGETS:
        panels = []
        counts = {domain: {region: np.zeros(4, dtype=np.int64) for region in REGIONS} for domain in DOMAINS}
        by_seed = {}
        for seed in SEEDS:
            path = work / f"block-{seed}-{budget}.json"
            row = json.loads(path.read_text())
            assert (row["seed"], row["budget"], row["gpu"], row["manifest_sha256"], row["checkpoint_sha256"]) == (
                seed, budget, profile["gpu"], manifest_hash, checkpoints[seed])
            assert row["layers"] > 0 and set(row["key_recovery"]) == set(DOMAINS)
            total = np.zeros(4, dtype=np.int64)
            for domain, n_inputs in DOMAINS.items():
                for region, size in zip(REGIONS, (16, 32, 16), strict=True):
                    saved = np.array(row["key_recovery"][domain][region], dtype=np.int64)
                    assert saved[3] == n_inputs * size * 4 * row["layers"]
                    recovery_table(saved)
                    counts[domain][region] += saved
                    total += saved
            by_seed[seed] = recovery_table(total)
            panels.append(score_matrix(row, book_ids, labels))
            hashes[path.stem] = digest(path)
        shortlist = quality(panels)
        positions = np.arange(4096)
        fraction = (np.minimum(budget - 1, positions // 64) * 64 + positions % 64 + 1).sum() / (positions + 1).sum()
        aggregate = sum((x for domain in counts.values() for x in domain.values()), np.zeros(4, dtype=np.int64))
        results[budget] = {"quality": rates(shortlist), "shortlist_minus_exact": comparison(shortlist, reference, families),
            "shortlist_minus_soft": comparison(shortlist, soft, families), "key_recovery": recovery_table(aggregate),
            "key_recovery_by_seed": by_seed, "key_recovery_by_domain": {
                d: {r: recovery_table(c) for r, c in regions.items()} for d, regions in counts.items()},
            "estimated_eligible_key_fraction_4k": float(fraction),
            "clean_mirrored_minus_exact": family_difference(shortlist["mirrored"][:, clean],
                                                            reference["mirrored"][:, clean], clean_families)}

    return {"manifest_sha256": manifest_hash, "previous_results_sha256": digest(previous_path),
            "previous_quality_sha256": previous_hashes, "source_audit_sha256": digest(audit_path),
            "raw_sha256": hashes, "gpu": profile["gpu"], "torch": profile["torch"], "triton": profile["triton"],
            "primary_budget": profile["primary_budget"], "fidelity": profile["checks"], "clean_questions": int(clean.sum()),
            "quality": {"exact": rates(reference), "soft": rates(soft), "shortlist": results},
            "full_prefill": summarize_timings(profile["full_prefill"], list(map(str, LENGTHS)), 15),
            "kernel": summarize_timings(profile["kernel"], [f"{k}-{n}" for k in ("prefill", "read-one") for n in LENGTHS], 30)}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "manifest", "previous", "source_audit", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    write_json(a.output, summarize(a.work, a.manifest, a.previous, a.source_audit))
