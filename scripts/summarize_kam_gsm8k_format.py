#!/usr/bin/env python3
"""Compare strict and conservative terminal scoring on the same GSM8K generation."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, FACTORIAL_RECIPES, FACTORIAL_RUNS, SEEDS, key
from summarize_kam_adapter_preservation import BASE, difference, read_archive


def summarize(replays, primary):
    read = read_archive(primary)
    replay = {key(*ids): json.loads((replays / f"format-gsm8k-{key(*ids)}.json").read_text())
              for ids in FACTORIAL_RUNS}
    assert len({v["metadata"]["source_sha256"] for v in replay.values()}) == 1
    by_fit, scored = {}, {}
    for ident, fit in replay.items():
        official = read(f"gsm8k-{ident}")
        for attr in ("source_sha256", "adapter_sha256", "model_fingerprints", "context_tokens", "gpu"):
            assert fit["metadata"][attr] == official["metadata"][attr], (ident, attr)
        rows = fit["records"]
        assert len(rows) == 1319
        methods = Counter(row["terminal_parser"] for row in rows)
        strict = np.array([r["correct"] for r in rows])
        parsed = np.array([r["parsed"] for r in rows])
        terminal = np.array([r["terminal_correct"] for r in rows])
        assert (strict <= terminal).all()
        assert np.array_equal(parsed, [row["terminal_parser"] == "marked" for row in rows])
        assert all(r["terminal_parser"] != "unrecognized" for r in rows if r["terminal_correct"])
        recorded = official["records"]
        assert [(r["index"], r["prompt_tokens"]) for r in rows] == [
            (r["index"], r["prompt_tokens"]) for r in recorded
        ]
        scores = {"strict": strict, "terminal": terminal}
        scored[ident] = scores
        by_fit[ident] = {
            "parsed_numeric": int(parsed.sum()),
            "strict_correct": int(strict.sum()),
            "terminal_correct": int(terminal.sum()),
            "original_strict_correct": sum(row["correct"] for row in recorded),
            "original_strict_or_parsed_flag_mismatches": sum(
                any(a[field] != b[field] for field in ("correct", "parsed"))
                for a, b in zip(rows, recorded, strict=True)
            ),
            "parser_counts": dict(methods),
            "newly_correct_by_parser": dict(Counter(r["terminal_parser"] for r in rows
                                                   if r["terminal_correct"] and not r["correct"])),
        }
    contrasts = {}
    for recipe in FACTORIAL_RECIPES:
        for mode in ("strict", "terminal"):
            outcomes = {arm: np.stack([scored[key(seed, recipe, arm)][mode] for seed in SEEDS])
                        for arm in ARMS}
            contrasts[f"{recipe}:{mode}:kam-minus-dot"] = difference(outcomes["kam"], outcomes["dot"])
            for arm in ARMS:
                contrasts[f"{recipe}:{mode}:{arm}-minus-untouched"] = difference(
                    outcomes[arm], scored[BASE][mode][None, :]
                )
    base = replay[BASE]["metadata"]
    return {"model": base["model"], "source_sha256": base["source_sha256"], "seed_order": SEEDS,
            "n": 1319, "per_checkpoint": by_fit, "contrasts": contrasts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", type=Path)
    parser.add_argument("primary_archive", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    a.output.write_text(json.dumps(summarize(a.replays, a.primary_archive), indent=2, sort_keys=True) + "\n")
