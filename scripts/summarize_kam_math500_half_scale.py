#!/usr/bin/env python3
"""Compare Llama full- and half-strength residuals on the same neutral prompts."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_kam_math500_repetition import repetitive
from kam_adapter_preservation import ARMS, LLAMA_ID, SEEDS, digest, key
from kam_math_kl_protocol import write_json
from summarize_kam_adapter_preservation import difference, read_archive

MEASURES = ("correct", "cap", "repetition_at_cap")


def enumerate_run(payload, tokens):
    rows = payload["records"]
    assert len(rows) == len(tokens)
    capped = np.array([r["stop"] == "budget" for r in rows])
    repeated = np.array(tokens, dtype=bool)
    flags = {"correct": np.array([r["full"]["correct"] for r in rows]), "cap": capped,
             "repetition_at_cap": capped & repeated}
    summary = {name: int(values.sum()) for name, values in flags.items()}
    summary.update({"seconds": payload["metadata"]["seconds"],
                    "mean_tokens": sum(row["output_tokens"] for row in rows) / len(rows)})
    return summary, flags


def summarize(half, original, raw_or_repetition, repetition_full, export_repetition=None):
    source = read_archive(original)
    full_repetition = json.loads(repetition_full.read_text())
    scaled_replay = json.loads(raw_or_repetition.read_text()) if raw_or_repetition.is_file() else None
    features = {}
    observations, vectors, hashes = {}, {}, {}
    n = None
    for seed in SEEDS:
        for arm in ARMS:
            name = key(seed, "lower_lr_kl05", arm)
            full = source(f"graded-{name}")
            scaled_path = half / f"graded-{name}.json"
            scaled = json.loads(scaled_path.read_text())
            a, b = full["metadata"], scaled["metadata"]
            assert a["model"] == b["model"] == LLAMA_ID
            assert (b["seed"], b["recipe"], b["arm"]) == (seed, "lower_lr_kl05", arm)
            assert b["protocol"] == a["protocol"].removesuffix("-v1") + "-half-scale-v1"
            assert b["residual_projection_scale"] == .5
            for field in ("source_sha256", "prompt_suffix", "selection_sha256", "adapter_sha256", "grader_protocol",
                          "grader_versions", "batch_size", "budget", "prefix_budget", "gpu", "training", "revision"):
                assert a[field] == b[field], (name, field)
            if "manifest_sha256" in a:
                assert a["manifest_sha256"] == b["manifest_sha256"]
            pairs = [r["id"] for r in full["records"]]
            assert pairs == [r["id"] for r in scaled["records"]]
            assert [r["prompt_tokens"] for r in full["records"]] == [r["prompt_tokens"] for r in scaled["records"]]
            n = len(pairs) if n is None else n
            assert len(pairs) == n
            archived = set(full_repetition["flag_ids"][name]["full"])
            full_tokens = [ident in archived for ident in pairs]
            if scaled_replay is None:
                raw_path = raw_or_repetition / f"neutral-{name}.json"
                assert digest(raw_path) == b["generation_sha256"]
                raw_rows = json.loads(raw_path.read_text())["records"]
                assert pairs == [r["id"] for r in raw_rows]
                repeated = [row["id"] for row in raw_rows if repetitive(row["completion"])]
                features[name] = {"generation_sha256": b["generation_sha256"], "ids": repeated}
            else:
                features[name] = scaled_replay[name]
                assert features[name]["generation_sha256"] == b["generation_sha256"]
            marked = set(features[name]["ids"])
            assert marked.issubset(pairs) and len(marked) == len(features[name]["ids"])
            half_tokens = [ident in marked for ident in pairs]
            observations[name], vectors[name] = {}, {}
            for group, payload, feature in (("full", full, full_tokens), ("half", scaled, half_tokens)):
                observations[name][group], vectors[name][group] = enumerate_run(payload, feature)
            hashes[name] = digest(scaled_path)
    contrasts = {}
    for stat in MEASURES:
        groups = {scale: {arm: np.stack([vectors[key(seed, "lower_lr_kl05", arm)][scale][stat] for seed in SEEDS])
                          for arm in ARMS} for scale in ("full", "half")}
        full, half = groups["full"], groups["half"]
        contrasts[stat] = {
            "full_kam_minus_dot": difference(full["kam"], full["dot"]),
            "half_kam_minus_dot": difference(half["kam"], half["dot"]),
            "change_in_kam_minus_dot": difference(half["kam"].astype(np.int8) + full["dot"],
                                                 half["dot"].astype(np.int8) + full["kam"]),
            "half_minus_full": {arm: difference(half[arm], full[arm]) for arm in ARMS},
        }
    if export_repetition is not None:
        write_json(export_repetition, features)
    return {"model": LLAMA_ID, "seeds": SEEDS, "n": n, "checkpoints": observations,
            "contrasts": contrasts, "scaled_outcome_sha256": hashes}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("half", type=Path)
    cli.add_argument("original", type=Path)
    cli.add_argument("raw_or_repetition", type=Path,
                     help="private generation folder or exported public repetition flags")
    cli.add_argument("repetition_full", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--export-repetition", type=Path)
    args = cli.parse_args()
    write_json(args.output, summarize(args.half, args.original, args.raw_or_repetition, args.repetition_full,
                                      args.export_repetition))
