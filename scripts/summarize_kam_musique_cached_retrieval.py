#!/usr/bin/env python3
"""Verify matched checkpoints and summarize fixed-deadline retrieval outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, digest
from kam_math_kl_protocol import write_json
from kam_musique_cached_retrieval import GPU, PROTOCOL, ROUTES
from summarize_kam_overnight_routes import timing, verified
from summarize_kam_pretrained_endpoint_selection import family_difference


def summarize(work, manifest, study):
    fixture = json.loads(manifest.read_text())
    assert fixture["gpu"] == GPU and fixture["deadline_ms"] == 60
    cached_identity, rank_reference = None, None
    checkpoints, hashes, joint_by_arm = {}, {}, {}
    for name, run, result, fingerprint in verified(work, manifest, study, "routes", PROTOCOL):
        seed, _, arm = run
        cache = result["metadata"]["cache"]
        cached_identity = cache if cached_identity is None else cached_identity
        assert cache == cached_identity and cache["manifest_sha256"] == digest(manifest)
        excluded = set(cache["excluded_pairs"])
        assert excluded <= {r["pair"] for r in fixture["cases"]}
        cases = [r for r in fixture["cases"] if r["pair"] not in excluded]
        expected = [(row["pair"], row["family"], q, reverse, i ^ reverse, slot, len(fixture["pool_ids"][q]))
                    for row in cases for i, q in enumerate(row["questions"])
                    for reverse in (False, True) for slot in (0, 1)]
        records = result["records"]
        header = ("pair", "family", "question", "reverse", "expected", "earlier_slot", "pool_size")
        assert [tuple(r[field] for field in header) for r in records] == expected
        assert len(expected) == 4 * cache["queries"]
        ranks = np.array([[row[method]["first_rank"] for method in ROUTES] for row in records])
        assert np.array_equal(ranks.reshape(-1, 4, 3)[:, :2], ranks.reshape(-1, 4, 3)[:, 2:])
        if rank_reference is None:
            rank_reference = ranks
        assert np.array_equal(ranks, rank_reference)
        checkpoints[name], hashes[name] = {}, fingerprint
        for split in sorted({row["split"] for row in cases}):
            index = {q for case in cases if case["split"] == split for q in case["questions"]}
            selected = [row for row in records if row["question"] in index]
            result = {}
            for method in ROUTES:
                values = [r[method] for r in selected]
                for row, v in zip(selected, values, strict=True):
                    assert 1 <= v["first_rank"] <= row["pool_size"] and v["first_correct"] == (v["first_rank"] == 1)
                    assert np.isfinite(v["margin"]) and np.isfinite(v["median_ms"]) and v["tokens"] > 0
                    assert 0 < v["retrieve_ms"] <= v["median_ms"]
                    assert v["correct"] == (v["margin"] != 0 and int(v["margin"] > 0) == row["expected"])
                first_ranks = np.array([v["first_rank"] for v in values]).reshape(-1, 4)
                endpoint = np.array([v["correct"] for v in values]).reshape(-1, 4)
                elapsed = np.array([v["median_ms"] for v in values]).reshape(-1, 4)
                joint = (first_ranks == 1) & endpoint
                eligible = (joint & (elapsed <= 60)).all(axis=1)
                joint_by_arm[arm, seed, method, split] = eligible
                result[method] = {"recall_at_1": int((first_ranks == 1).all(axis=1).sum()),
                                  "recall_at_3": int((first_ranks <= 3).all(axis=1).sum()),
                                  "endpoint": int(endpoint.all(axis=1).sum()), "joint": int(joint.all(axis=1).sum()),
                                  "joint_within_60ms": int(eligible.sum()), **timing(elapsed.flatten()),
                                  "retrieval_p50_ms": float(np.median([v["retrieve_ms"] for v in values]))}
            checkpoints[name][split] = result
    strata = {}
    for split in sorted({case["split"] for case in cases}):
        cohort = [c for c in cases if c["split"] == split]
        families = [c["family"] for c in cohort for _ in c["questions"]]
        total = len(families)
        metrics = {}
        for method in ROUTES:
            ref = next(iter(checkpoints.values()))[split][method]
            metrics[method] = {"first_hop_recall": {str(k): ref[f"recall_at_{k}"] / total for k in (1, 3)}, "arms": {}}
            for arm in (*ARMS, "pretrained"):
                seeds = (0,) if arm == "pretrained" else SEEDS
                rows = [r[split][method] for name, r in checkpoints.items() if name.endswith("-" + arm)]
                measured = {stat: float(np.mean([r[stat] for r in rows])) for stat in
                            ("p50_ms", "p95_ms", "retrieval_p50_ms")}
                for stat in ("joint", "joint_within_60ms", "endpoint"):
                    measured[stat] = float(np.mean([r[stat] for r in rows]) / total)
                measured["against_bm25_at_60ms"] = family_difference(
                    np.stack([joint_by_arm[arm, seed, method, split] for seed in seeds]),
                    np.stack([joint_by_arm[arm, seed, "bm25", split] for seed in seeds]), families)
                metrics[method]["arms"][arm] = measured
            paired = {arm: np.stack([joint_by_arm[arm, seed, method, split] for seed in SEEDS]) for arm in ARMS}
            untouched = np.broadcast_to(joint_by_arm["pretrained", 0, method, split], paired["kam"].shape)
            metrics[method]["between_arms_at_60ms"] = {
                "kam_minus_dot": family_difference(paired["kam"], paired["dot"], families),
                "kam_minus_untouched": family_difference(paired["kam"], untouched, families),
                "dot_minus_untouched": family_difference(paired["dot"], untouched, families),
            }
        strata[split] = {"questions": total, "families": len(set(families)),
                         "questions_in_prior_paired_families": sum(len(c["questions"]) for c in cohort if c["prior_family"]),
                         "pool_sizes": Counter(len(fixture["pool_ids"][q]) for c in cohort for q in c["questions"]),
                         "metrics": metrics}
    return {"manifest_sha256": digest(manifest), "outcome_sha256": hashes, "cache": cached_identity,
            "checkpoints": checkpoints, "strata": strata}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "manifest", "study", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    write_json(args.output, summarize(args.work, args.manifest, args.study))
