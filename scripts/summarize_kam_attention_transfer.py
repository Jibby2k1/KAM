#!/usr/bin/env python3
"""Verify independent 4K/8K transfer and compare paired quality and full prefill."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from prepare_kam_attention_transfer import PROTOCOL, QUESTIONS, WIDTHS
from summarize_kam_block_shortlist import comparison, quality, rates, score_matrix
from summarize_kam_live_token_factorial import estimate
from summarize_kam_pretrained_endpoint_selection import family_difference

ARMS = ("flash", "protected", "selected")


def read(path):
    return json.loads(path.read_text())


def summarize(work, fixture_path, plan_path, audit_path, backbone, previous):
    fixture, plan, audit, old = map(read, (fixture_path, plan_path, audit_path, previous))
    assert fixture["protocol"] == PROTOCOL and fixture["scored_tokens_per_window"] == 1024
    assert plan["transfer_arm"] == old["transfer_arm"] and plan["source_summary_sha256"] == digest(previous)
    fixture_hash, plan_hash = map(digest, (fixture_path, plan_path))
    with gzip.open(backbone, "rt") as stream:
        prior = json.load(stream)
    checks = {s: json.loads(prior[f"backbone-{s}-0"])["checkpoint_sha256"] for s in SEEDS}
    profile_path = work / "transfer-profile.json"
    profile = read(profile_path)
    gpu = profile["gpu"]
    assert (profile["seed"], profile["checkpoint_sha256"], profile["transfer_manifest_sha256"],
            profile["selection_sha256"], profile["selected"]) == (SEEDS[0], checks[SEEDS[0]], fixture_hash, plan_hash, plan["transfer_arm"])
    hashes = {profile_path.stem: digest(profile_path)}
    book_ids = [r["id"] for r in fixture["pg19"]]
    assert len(book_ids) + len(fixture["short_books"]) == 28
    overlaps = {hit for source in audit.values() for hit in source["matches"]["test"]}
    by_width = {}
    scores, timing, clean_counts = {}, {}, {}
    family_order = None
    for width in map(str, WIDTHS):
        pairs = fixture["fixture"][width]["questions"]
        labels = [(r["id"], r["family"], order) for r in pairs for order in (0, 1)]
        families = [x["family"] for x in pairs]
        if family_order is None:
            family_order = families
            matched_labels = labels
        assert labels == matched_labels and families == family_order and len(pairs) == QUESTIONS
        clean = np.array([not overlaps.intersection([*r["support_ids"], r["foil_id"]]) for r in pairs])
        clean_counts[width] = int(clean.sum())
        records = {}
        for arm in ARMS:
            rows = []
            for seed in SEEDS:
                p = work / f"transfer-{seed}-{arm}.json"
                result = read(p)
                hashes[p.stem] = digest(p)
                assert (result["seed"], result["arm"], result["gpu"], result["checkpoint_sha256"]) == (seed, arm, gpu, checks[seed])
                assert (result["transfer_manifest_sha256"], result["selection_sha256"], result["selected"]) == (
                    fixture_hash, plan_hash, plan["transfer_arm"])
                rows.append(score_matrix(result["outcomes"][width], book_ids, labels, windows=1))
            records[arm] = quality(rows)
        by_width[width] = records
        scores[width] = {"arms": {arm: rates(r) for arm, r in records.items()},
                         "selected_minus": {arm: comparison(records["selected"], records[arm], families)
                                            for arm in ("flash", "protected")},
                         "clean_mirrored_vs_flash": family_difference(records["selected"]["mirrored"][:, clean],
                             records["flash"]["mirrored"][:, clean], [f for f, keep in zip(families, clean, strict=True) if keep])}
        timing[width] = {}
        assert set(profile["full_prefill"][width]) == set(ARMS)
        for arm, row in profile["full_prefill"][width].items():
            wall, event = map(np.asarray, (row["wall_ms"], row["cuda_ms"]))
            assert len(wall) == len(event) == 15 and all((v > 0).all() and np.isfinite(v).all() for v in (wall, event))
            timing[width][arm] = {"wall_p50_ms": float(np.median(wall)), "wall_p95_ms": float(np.quantile(wall, .95)),
                                  "cuda_p50_ms": float(np.median(event)), "peak_bytes": row["incremental_allocated_bytes"]}

    lo, hi = (by_width[str(width)] for width in WIDTHS)
    interaction = {}
    for reference in ("flash", "protected"):
        nll = [panel["selected"]["nll"] - panel[reference]["nll"] for panel in (hi, lo)]
        mirror = [panel["selected"]["mirrored"].astype(np.int8) - panel[reference]["mirrored"].astype(np.int8)
                  for panel in (hi, lo)]
        interaction[reference] = {"nll": estimate(nll[0]-nll[1]),
                                   "mirrored": family_difference(mirror[0], mirror[1], family_order)}
    return {"manifest_sha256": fixture_hash, "selection_sha256": plan_hash, "source_audit_sha256": digest(audit_path),
            "raw_sha256": hashes, "gpu": gpu, "selected": plan["transfer_arm"], "book_count": len(book_ids),
            "question_count": QUESTIONS, "clean_questions": clean_counts, "scores": scores, "timings": timing,
            "relative_gap_change_at_8k": interaction}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "fixture", "plan", "audit", "backbone", "previous", "summary"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    write_json(a.summary, summarize(a.work, a.fixture, a.plan, a.audit, a.backbone, a.previous))
