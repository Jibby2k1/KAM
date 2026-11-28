#!/usr/bin/env python3
"""Reproduce the frozen TabMWP comparisons from paired numeric outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, LLAMA_ID, SEEDS, digest, key
from kam_math_kl_protocol import test_runs, write_json
from kam_tabmwp_archive_noise import BUDGET, MODES, PROTOCOL
from summarize_kam_adapter_preservation import read_archive
from summarize_kam_pretrained_endpoint_selection import family_difference


def summarize(work, manifest_path, selected_dir):
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"] == PROTOCOL and tuple(manifest["modes"]) == MODES
    selection_path = selected_dir / "llama3-selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["model"] == LLAMA_ID and selection["selected_kl_weight"] == .5
    recipe = selection["selected_recipe"]
    runs = test_runs(recipe)
    prior = read_archive(selected_dir / "llama3-outcomes.json.gz")
    expected = [tuple(row[k] for k in ("id", "grade", "family")) for row in manifest["cases"]]
    families = [row[2] for row in expected]
    phases, measures, hashes = {}, {}, {}
    layout = scoring_window = None
    for seed, trained, arm in runs:
        ident = key(seed, trained, arm)
        item = work / f"tabmwp-{ident}.json"
        doc = json.loads(item.read_text())
        meta, rows = doc["metadata"], doc["records"]
        window = meta.get("scoring_window", "full")
        assert window in ("full", "terminal_three_lines")
        scoring_window = window if scoring_window is None else scoring_window
        assert window == scoring_window
        assert (meta["seed"], meta["recipe"], meta["arm"]) == (seed, trained, arm)
        assert meta["protocol"] == PROTOCOL and meta["manifest_sha256"] == digest(manifest_path)
        reference = prior(f"test-{ident}")["metadata"]
        assert meta["selection_sha256"] == reference["selection_sha256"] == digest(selection_path)
        for field in ("model", "gpu", "revision", "model_fingerprints", "adapter_sha256", "training"):
            assert meta[field] == reference[field], field
        assert [tuple(r[k] for k in ("id", "grade", "family")) for r in rows] == expected
        widths = [tuple(row[mode]["prompt_tokens"] for mode in MODES) for row in rows]
        assert all(0 < a < b == c for a, b, c in widths)
        layout = widths if layout is None else layout
        assert layout == widths
        phases[ident], measures[ident] = {}, {}
        for mode in MODES:
            scored = [row[mode] for row in rows]
            assert all(set(r) == {"correct", "parsed", "prompt_tokens", "output_tokens", "stop", "issues"}
                       for r in scored)
            assert all(r["correct"] <= r["parsed"] and 0 <= r["output_tokens"] <= BUDGET for r in scored)
            assert all((r["stop"], r["output_tokens"] == BUDGET) in (("budget", True), ("eos", False))
                       for r in scored)
            phases[ident][mode] = np.array([r["correct"] for r in scored], dtype=bool)
            measures[ident][mode] = {
                "correct": int(phases[ident][mode].sum()),
                "parsed": sum(r["parsed"] for r in scored),
                "cap": sum(r["stop"] == "budget" for r in scored),
                "issues": sum(len(r["issues"]) for r in scored),
                "seconds": meta["seconds"][mode],
            }
        hashes[ident] = digest(item)
    base = phases[key(*runs[-1])]
    flags = {mode: {arm: np.stack([phases[key(seed, recipe, arm)][mode] for seed in SEEDS])
                    for arm in ARMS}
             for mode in MODES}
    by_mode = {mode: {"untouched": float(base[mode].mean()), "means": {arm: float(p.mean()) for arm, p in fits.items()},
                      "kam_minus_dot": family_difference(fits["kam"], fits["dot"], families)}
               for mode, fits in flags.items()}
    comparisons = (("archive_last", "current_only"), ("archive_first", "current_only"), ("archive_last", "archive_first"))
    contrasts = {}
    for after, before in comparisons:
        a, b = flags[after], flags[before]
        contrasts[f"{after}_minus_{before}"] = {
            "untouched_change": float(base[after].mean() - base[before].mean()),
            "arm_changes": {arm: family_difference(a[arm], b[arm], families) for arm in ARMS},
            "difference_in_differences": family_difference(
                a["kam"].astype(np.int8) + b["dot"], a["dot"].astype(np.int8) + b["kam"], families
            ),
            "net_transitions": {arm: {"gained": (a[arm] & ~b[arm]).sum(axis=1).tolist(),
                                      "lost": (b[arm] & ~a[arm]).sum(axis=1).tolist()} for arm in ARMS},
        }
    return {"n": len(expected), "families": len(set(families)), "seeds": SEEDS, "checkpoints": measures,
            "accuracy": by_mode, "contrasts": contrasts, "manifest_sha256": digest(manifest_path),
            "outcome_sha256": hashes,
            **({"scoring_window": scoring_window} if scoring_window != "full" else {})}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "manifest", "selected_dir", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    write_json(args.output, summarize(args.work, args.manifest, args.selected_dir))
