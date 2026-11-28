#!/usr/bin/env python3
"""Cross-fit an inference-visible, cost-aware Flash rescue ranker."""
import argparse
from pathlib import Path

import numpy as np

from analyze_kam_confidence_deferral import FIRST, archive, choose, correct, oracle, read, records, weights
from audit_kam_natural_data import stable
from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from summarize_kam_distant_ablation import continuous_interval
from summarize_kam_pretrained_endpoint_selection import family_difference

PROTOCOL = "kam-quality-rescue-ranking-v1"
FOLDS = 5
RIDGE = 0.10
BUDGET = 0.99
FEATURES = ("top_two_gap", "entropy", "log_tokens", "slot_a", "slot_b", "slot_c", "slot_d")


def features(logits, lengths):
    ordered = np.sort(logits, axis=-1)
    gap = ordered[..., -1] - ordered[..., -2]
    centered = logits - ordered[..., -1:]
    masses = np.exp(centered)
    probabilities = masses / masses.sum(axis=-1, keepdims=True)
    log_probabilities = centered - np.log(masses.sum(axis=-1, keepdims=True))
    entropy = -(probabilities * log_probabilities).sum(axis=-1)
    tied = centered == 0
    winner = tied / tied.sum(axis=-1, keepdims=True)
    scalar = np.stack((gap, entropy, np.broadcast_to(np.log(lengths), gap.shape)), axis=-1)
    return gap, np.concatenate((scalar, winner), axis=-1)


def predict(design, gain, fit):
    training = design[:, fit].reshape(-1, design.shape[-1])
    target = gain[:, fit].reshape(-1)
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (training - mean) / scale
    centered_target = target - target.mean()
    size = target.size
    coefficients = np.linalg.solve(
        standardized.T @ standardized / size + RIDGE * np.eye(training.shape[1]),
        standardized.T @ centered_target / size,
    )
    prediction = (design - mean) / scale @ coefficients + target.mean()
    return prediction, {"mean": mean.tolist(), "scale": scale.tolist(),
                        "intercept": float(target.mean()), "coefficients": coefficients.tolist()}


def describe(picks, fast, flash, first_ms, flash_ms, ids):
    answer = np.where(picks, flash, fast)
    cost = first_ms + picks * flash_ms
    return {
        "correct": int(answer.sum()),
        "correct_by_seed": answer.sum(axis=1).tolist(),
        "accuracy": float(answer.mean()),
        "defer_rate": float(picks.mean()),
        "rescues": int((picks & flash & ~fast).sum()),
        "regressions": int((picks & ~flash & fast).sum()),
        "unchanged": int((picks & (flash == fast)).sum()),
        "mean_predicted_ms": float(cost.mean()),
        "paired_accuracy_vs_flash": family_difference(answer, flash, ids),
        "paired_accuracy_vs_no_rerun": family_difference(answer, fast, ids),
        "paired_ms_vs_flash": continuous_interval(cost - flash_ms, ids),
    }


def compare(logits, gold, flash, first_ms, flash_ms, ids, folds, lengths):
    fast = correct(logits, gold)
    gain = flash.astype(np.int8) - fast.astype(np.int8)
    gap, design = features(logits, lengths)
    decisions = {name: np.zeros_like(fast) for name in ("no_rerun", "margin", "learned")}
    fit_artifacts = []
    for fold in range(FOLDS):
        test = folds == fold
        fit = ~test
        prediction, params = predict(design, gain, fit)
        # Sort high predicted rescue per millisecond first. Exclude zero or negative gains.
        ranking = np.where(prediction > 0, -prediction * 1000 / flash_ms, np.inf)
        tuning = {}
        for name, scores in (("margin", gap), ("learned", ranking)):
            cutoff, fit_summary = choose(scores[:, fit], fast[:, fit], flash[:, fit], first_ms[fit],
                                         flash_ms[fit], budget_fraction=BUDGET)
            if cutoff is not None:
                decisions[name][:, test] = scores[:, test] <= cutoff
            tuning[name] = {"cutoff": cutoff, **fit_summary}
        fit_artifacts.append({"fold": fold, "heldout_article_ids": sorted(set(ids[test])),
                              "heldout_questions": int(test.sum()), "ridge_parameters": params,
                              "cutoff_selection": tuning})
    decisions["oracle"], _ = oracle(fast, flash, first_ms, flash_ms, budget_fraction=BUDGET)
    outcomes = {name: describe(picks, fast, flash, first_ms, flash_ms, ids)
                for name, picks in decisions.items()}
    learned_answer = np.where(decisions["learned"], flash, fast)
    margin_answer = np.where(decisions["margin"], flash, fast)
    outcomes["learned"]["paired_accuracy_vs_margin"] = family_difference(
        learned_answer, margin_answer, ids)
    return {"flash_correct": int(flash.sum()), "flash_accuracy": float(flash.mean()),
            "flash_mean_ms": float(flash_ms.mean()), "methods": outcomes, "folds": fit_artifacts}


def summary(manifest_path, summary_path, archive_path):
    manifest, frozen_summary = map(read, (manifest_path, summary_path))
    assert frozen_summary["manifest_sha256"] == digest(manifest_path)
    rows = manifest["selected"]
    ids = np.array([row["article_id"] for row in rows])
    ordered_articles = sorted(set(ids), key=lambda article: stable(PROTOCOL + str(article)))
    assign_fold = {article: i % FOLDS for i, article in enumerate(ordered_articles)}
    folds = np.array([assign_fold[article] for article in ids])
    assert len(ordered_articles) == 113 and len(rows) == 452
    gold = np.array([row["answer"] for row in rows])
    lengths = np.array([row["lengths"][0] for row in rows])
    content = archive(archive_path)
    flash = correct(records(content, rows, "flash"), gold)
    results = {}
    for arm in FIRST:
        timing = frozen_summary["budget_timing"] if arm == "output-selected" else frozen_summary["timing"]
        first_ms, flash_ms = (weights(timing, policy, lengths) for policy in (arm, "flash"))
        logits = records(content, rows, arm)
        results[arm] = compare(logits, gold, flash, first_ms, flash_ms, ids, folds, lengths)
    return {"protocol": PROTOCOL, "articles": len(ordered_articles), "questions": len(rows),
            "seeds": SEEDS, "ranker_features": FEATURES, "standardized_ridge_penalty": RIDGE,
            "budget_fraction": BUDGET, "manifest_sha256": digest(manifest_path),
            "summary_sha256": digest(summary_path), "outcomes_sha256": digest(archive_path),
            "first_passes": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "summary", "archive", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write_json(args.output, summary(args.manifest, args.summary, args.archive))
