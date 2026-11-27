#!/usr/bin/env python3
"""Verify the GSM8K decoding budget and SVAMP arithmetic transfer experiments."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, key
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


def compare_svamp(root, primary):
    payloads = {ident: json.loads((root / f"svamp-{ident}.json").read_text()) for ident in RUNS}
    base = payloads[BASE]
    ids = [(r["id"], r["type"], r["prompt_tokens"]) for r in base["records"]]
    assert len(ids) == len({row[0] for row in ids}) == 1000
    vectors, alongside = {}, {}
    for ident, payload in payloads.items():
        control = primary(f"gsm8k-{ident}")["metadata"]
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
        metrics[kind] = {"n": int(keep.sum()), "correct": counts, "parsed": parsed, "contrasts": comparison}
    return {"task": "svamp", "source": {k: base["metadata"][k] for k in ("source_revision", "source_sha256", "generation_budget")},
            "model": base["metadata"]["model"], "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("budget", "svamp"))
    parser.add_argument("input", type=Path)
    parser.add_argument("primary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize = {"budget": compare_budget, "svamp": compare_svamp}[args.mode]
    outcome = summarize(args.input, read_archive(args.primary))
    args.output.write_text(json.dumps(outcome, sort_keys=True, indent=2) + "\n")
