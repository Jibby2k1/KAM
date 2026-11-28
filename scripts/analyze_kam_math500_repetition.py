#!/usr/bin/env python3
"""Audit a fixed lexical repetition probe on archived marker-free generations."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, digest, key
from kam_math_kl_protocol import test_runs, write_json
from kam_math_repetition import NGRAM, RATE, WINDOW, repetitive
from summarize_kam_adapter_preservation import difference, read_archive

def summarize(raw, scored, selection):
    selected = json.loads(selection.read_text())
    recipe = selected["selected_recipe"]
    reference = read_archive(scored)
    per_run, vectors, indices = {}, {}, {}
    for run in test_runs(recipe):
        name = key(*run)
        path = raw / f"neutral-{name}.json"
        payload = json.loads(path.read_text())
        grade = reference(f"graded-{name}")
        assert digest(path) == grade["metadata"]["generation_sha256"]
        rows, grades = payload["records"], grade["records"]
        assert len(rows) == len(grades)
        assert [row["id"] for row in rows] == [row["id"] for row in grades]
        capped = np.array([r["stop"] == "budget" for r in rows])
        full = np.array([repetitive(r["completion"]) for r in rows])
        prefix = np.array([repetitive(r["prefix"]) for r in rows])
        correct = np.array([r["full"]["correct"] for r in grades])
        categories = {"capped": capped, "repetitive_at_cap": full & capped,
                      "repetitive_at_prefix_and_cap": prefix & capped,
                      "repetitive_at_prefix_then_eos": prefix & ~capped,
                      "repetitive_then_eos": full & ~capped,
                      "score_on_early_repetitive": prefix & correct}
        per_run[name] = {label: int(flags.sum()) for label, flags in categories.items()}
        vectors[name] = categories
        indices[name] = {label: [row["id"] for row, keep in zip(rows, flags, strict=True) if keep]
                         for label, flags in (("full", full), ("prefix", prefix))}
    contrasts = {}
    for mode in next(iter(vectors.values())):
        a, b = (np.stack([vectors[key(seed, recipe, arm)][mode] for seed in SEEDS]) for arm in ARMS)
        contrasts[mode] = difference(a, b)
    return {"model": selected["model"], "window": WINDOW, "ngram": NGRAM, "rate": RATE,
            "per_checkpoint": per_run, "flag_ids": indices, "contrasts": contrasts}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw", type=Path)
    cli.add_argument("scores", type=Path)
    cli.add_argument("selection", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    write_json(args.output, summarize(args.raw, args.scores, args.selection))
