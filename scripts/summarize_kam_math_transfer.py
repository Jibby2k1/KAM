#!/usr/bin/env python3
"""Verify the GSM8K decoding budget and SVAMP arithmetic transfer experiments."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, LOW_RATE_KL_RECIPE, LOW_RATE_KL_RUNS, SEEDS, key
from summarize_kam_adapter_preservation import BASE, difference, read_archive

RUNS = [BASE, *(key(seed, "lower_lr", arm) for seed in SEEDS for arm in ARMS)]


def compare_budget(root, primary):
    scores, vectors = {}, {}
    for ident in RUNS:
        audition = json.loads((root / f"gsm8k-budget-{ident}.json").read_text())
        control = primary(f"gsm8k-{ident}")
        assert audition["source_sha256"] == control["metadata"]["source_sha256"]
        assert audition["checkpoint"]["adapter_sha256"] == control["metadata"]["adapter_sha256"]
        assert (audition["short_budget"], audition["long_budget"]) == (384, 768)
        rows = audition["records"]
        assert len(rows) == 1319
        for row, original in zip(rows, control["records"], strict=True):
            assert row["replay_agrees"] and row["index"] == original["index"]
            assert row["short_correct"] == original["correct"] and row["short_parsed"] == original["parsed"]
        scores[ident] = {"n": len(rows), "stop": dict(Counter(r["long_stop"] for r in rows)),
                         "continued_past_short_limit": sum(r["output_tokens"] > 384 for r in rows)}
        vectors[ident] = {}
        for mode in ("short", "long"):
            vectors[ident][mode] = np.asarray([row[f"{mode}_correct"] for row in rows])
            scores[ident][f"{mode}_correct"] = int(vectors[ident][mode].sum())
            scores[ident][f"{mode}_parsed"] = sum(row[f"{mode}_parsed"] for row in rows)
    effects = {}
    for arm in ARMS:
        for mode in ("short", "long"):
            measured = np.stack([vectors[key(seed, "lower_lr", arm)][mode] for seed in SEEDS])
            effects[f"{mode}:{arm}-minus-untouched"] = difference(measured, vectors[BASE][mode][None, :])
    return {"task": "gsm8k", "per_checkpoint": scores, "contrasts": effects}


def compare_svamp(root, primary, pilot=None):
    pilot_keys = [key(*run) for run in LOW_RATE_KL_RUNS] if pilot else []
    payloads = {ident: json.loads((root / f"svamp-{ident}.json").read_text()) for ident in (*RUNS, *pilot_keys)}
    base = payloads[BASE]
    ids = [(r["id"], r["type"], r["prompt_tokens"]) for r in base["records"]]
    assert len(ids) == len({row[0] for row in ids}) == 1000
    vectors, alongside = {}, {}
    for ident, payload in payloads.items():
        read = pilot if ident in pilot_keys else primary
        control = read(f"gsm8k-{ident}")["metadata"]
        for field in ("model", "revision", "adapter_sha256", "model_fingerprints", "gpu"):
            assert payload["metadata"][field] == control[field]
        for field in ("source_revision", "source_sha256", "generation_budget"):
            assert payload["metadata"][field] == base["metadata"][field]
        rows = payload["records"]
        assert ids == [(r["id"], r["type"], r["prompt_tokens"]) for r in rows]
        assert all(r["correct"] <= r["parsed"] for r in rows)
        vectors[ident] = np.asarray([r["correct"] for r in rows])
        alongside[ident] = np.asarray([r["parsed"] for r in rows])
    types = np.asarray([row[1] for row in ids])
    metrics = {}
    for kind in ["all", *sorted(set(types))]:
        keep = np.ones(len(ids), dtype=bool) if kind == "all" else types == kind
        counts = {ident: int(values[keep].sum()) for ident, values in vectors.items()}
        parsed = {ident: int(values[keep].sum()) for ident, values in alongside.items()}
        per_arm = {arm: np.stack([vectors[key(seed, "lower_lr", arm)][keep] for seed in SEEDS]) for arm in ARMS}
        comparison = {f"{arm}-minus-untouched": difference(outcomes, vectors[BASE][None, keep])
                      for arm, outcomes in per_arm.items()}
        comparison["kam-minus-dot"] = difference(per_arm["kam"], per_arm["dot"])
        if pilot:
            moderate = {arm: np.stack([vectors[key(seed, LOW_RATE_KL_RECIPE, arm)][keep] for seed in SEEDS])
                        for arm in ARMS}
            comparison["kl05:kam-minus-dot"] = difference(moderate["kam"], moderate["dot"])
            for arm in ARMS:
                comparison[f"kl05:{arm}-minus-untouched"] = difference(moderate[arm], vectors[BASE][None, keep])
                comparison[f"kl05:{arm}-minus-no-kl"] = difference(moderate[arm], per_arm[arm])
        metrics[kind] = {"n": int(keep.sum()), "correct": counts, "parsed": parsed, "contrasts": comparison}
    return {"task": "svamp", "source": {k: base["metadata"][k] for k in ("source_revision", "source_sha256", "generation_budget")},
            "model": base["metadata"]["model"], "metrics": metrics}


def compare_scaling(root, primary):
    results, labels, vectors = {}, None, {}
    for ident in RUNS:
        result = json.loads((root / f"svamp-scaling-{ident}.json").read_text())
        meta = result["metadata"]
        reference = primary(f"gsm8k-{ident}")["metadata"]
        assert meta["adapter_sha256"] == reference["adapter_sha256"]
        assert meta["model_fingerprints"] == reference["model_fingerprints"]
        rows = result["records"]
        observed = [(r["index"], r["id"], r["type"], r["multiplier"]) for r in rows]
        assert len(observed) == len({r[1] for r in observed}) == 149
        labels = observed if labels is None else labels
        assert observed == labels
        assert all(not (r["scaled_correct"] and r["scaled_selected_original_answer"]) for r in rows)
        assert all(r["original_correct"] <= r["original_parsed"] and
                   r["scaled_correct"] <= r["scaled_parsed"] for r in rows)
        flags = {mode: np.asarray([r[f"{mode}_correct"] for r in rows])
                 for mode in ("original", "scaled")}
        flags["both"] = flags["original"] & flags["scaled"]
        vectors[ident] = flags
        results[ident] = {f"{mode}_correct": int(v.sum()) for mode, v in flags.items()}
        for mode in ("original", "scaled"):
            results[ident][f"{mode}_marked"] = sum(row[f"{mode}_parsed"] for row in rows)
        results[ident]["scaled_with_original_answer"] = sum(r["scaled_selected_original_answer"] for r in rows)
    comparisons = {}
    for arm in ARMS:
        scored = {mode: np.stack([vectors[key(seed, "lower_lr", arm)][mode] for seed in SEEDS])
                  for mode in ("original", "scaled", "both")}
        comparisons[f"{arm}:scaled-minus-original"] = difference(scored["scaled"], scored["original"])
        for mode in ("original", "scaled", "both"):
            comparisons[f"{arm}:{mode}-minus-untouched"] = difference(scored[mode], vectors[BASE][mode][None, :])
    return {"task": "svamp-scaling", "n": len(labels), "types": dict(Counter(row[2] for row in labels)),
            "scale_factors": dict(Counter(row[3] for row in labels)), "per_checkpoint": results,
            "contrasts": comparisons}


def compare_answer_format(root, official):
    outcomes, flags = {}, {}
    for ident in (*RUNS, *(key(*run) for run in LOW_RATE_KL_RUNS)):
        original = official(f"svamp-{ident}")
        alternate = json.loads((root / f"svamp-answer-format-{ident}.json").read_text())
        for field in ("model", "revision", "adapter_sha256", "model_fingerprints", "gpu", "source_sha256",
                      "source_revision", "generation_budget"):
            assert alternate["metadata"][field] == original["metadata"][field]
        rows = alternate["records"]
        assert len(rows) == 1000
        assert [(r["id"], r["type"]) for r in original["records"]] == [(r["id"], r["type"]) for r in rows]
        old = np.array([row["correct"] for row in original["records"]])
        new = np.array([row["correct"] for row in rows])
        any_format = np.array([row["any_marker_correct"] for row in rows])
        assert (new <= any_format).all()
        flags[ident] = {"hash": old, "answer": new, "either": any_format, "both_requested": old & new}
        outcomes[ident] = {"marked_hash_format": sum(r["parsed"] for r in original["records"]),
                           "marked_answer_format": sum(r["parsed"] for r in rows),
                           "old_marker_in_answer_format": sum(r["legacy_marker_parsed"] for r in rows),
                           "any_marker_in_answer_format": sum(r["any_marker_parsed"] for r in rows),
                           **{mode: int(values.sum()) for mode, values in flags[ident].items()}}
    comparisons = {}
    for recipe in ("lower_lr", LOW_RATE_KL_RECIPE):
        for arm in ARMS:
            scored = {mode: np.stack([flags[key(seed, recipe, arm)][mode] for seed in SEEDS])
                      for mode in ("hash", "answer", "either", "both_requested")}
            comparisons[f"{recipe}:{arm}:answer-minus-hash"] = difference(scored["answer"], scored["hash"])
            for mode in ("answer", "either", "both_requested"):
                comparisons[f"{recipe}:{arm}:{mode}-minus-untouched"] = difference(scored[mode], flags[BASE][mode][None, :])
    return {"task": "svamp-answer-format", "n": 1000, "per_checkpoint": outcomes, "contrasts": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("budget", "svamp", "svamp-scaling", "svamp-answer-format"))
    parser.add_argument("input", type=Path)
    parser.add_argument("primary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    primary = read_archive(args.primary)
    if args.mode == "svamp":
        pilot = read_archive(args.pilot) if args.pilot else None
        outcome = compare_svamp(args.input, primary, pilot)
    elif args.mode == "budget":
        outcome = compare_budget(args.input, primary)
    elif args.mode == "svamp-scaling":
        outcome = compare_scaling(args.input, primary)
    else:
        outcome = compare_answer_format(args.input, primary)
    args.output.write_text(json.dumps(outcome, sort_keys=True, indent=2) + "\n")
