#!/usr/bin/env python3
"""Summarize paired effects of prompt requests and grading on the same questions."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, digest, key
from kam_gsm8k_format_effects import PROMPTS, PROTOCOL
from kam_math_kl_protocol import BUDGET, MARKERS, test_runs, write_json
from summarize_kam_adapter_preservation import difference, read_archive

CELLS = tuple((mode, "symbolic") for mode in PROMPTS) + tuple((mode, "strict") for mode in MARKERS)


def flags(rows, attribute):
    return {f"{mode}:{grader}": np.array([row[mode][grader][attribute] for row in rows])
            for mode, grader in CELLS}


def gap_change(old, new):
    return difference(new["kam"].astype(np.int8) + old["dot"],
                      new["dot"].astype(np.int8) + old["kam"])


def summarize(work, model, manifest_path, selection_dir, gold_audit):
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"] == PROTOCOL and manifest["prompts"] == PROMPTS
    assert manifest["budget"] == BUDGET and len(manifest["ids"]) == 512
    selection_path = selection_dir / f"{model}-selection.json"
    selection = json.loads(selection_path.read_text())
    runs = test_runs(selection["selected_recipe"])
    original = read_archive(selection_dir / f"{model}-outcomes.json.gz")
    grader = json.loads(gold_audit.read_text())
    outcomes = {"correct": {}, "parsed": {}}
    stats, hashes = {}, {}
    prompt_lengths = None
    for run in runs:
        ident = key(*run)
        file = work / f"scored-{ident}.json"
        scores = json.loads(file.read_text())
        meta, rows = scores["metadata"], scores["records"]
        old_meta = original(f"test-{ident}")["metadata"]
        assert meta["protocol"] == PROTOCOL and meta["model"] == selection["model"]
        assert (meta["seed"], meta["recipe"], meta["arm"]) == run
        assert meta["selection_sha256"] == old_meta["selection_sha256"] == digest(selection_path)
        assert meta["manifest_sha256"] == digest(manifest_path)
        assert meta["fixture_sha256"] == manifest["fixture_sha256"]
        assert (meta["grader_protocol"], meta["grader_versions"]) == (grader["protocol"], grader["versions"])
        for field in ("adapter_sha256", "model_fingerprints", "training", "revision", "gpu", "new_trainable_parameters"):
            assert meta[field] == old_meta[field], (ident, field)
        assert [row["id"] for row in rows] == manifest["ids"]
        lengths = [tuple(row[m]["prompt_tokens"] for m in PROMPTS) for row in rows]
        prompt_lengths = lengths if prompt_lengths is None else prompt_lengths
        assert lengths == prompt_lengths
        for row in rows:
            assert set(row) == {"id", *PROMPTS}
            for mode, result in ((mode, row[mode]) for mode in PROMPTS):
                expected = {"prompt_tokens", "output_tokens", "stop", "symbolic"}
                if mode in MARKERS:
                    expected.update(("strict", "marker_present"))
                assert set(result) == expected
                assert 0 <= result["output_tokens"] <= BUDGET and result["prompt_tokens"] > 0
                assert (result["stop"], result["output_tokens"] == BUDGET) in (("budget", True), ("eos", False))
            for mode, grader_name in CELLS:
                cell = row[mode][grader_name]
                assert set(cell) == {"parsed", "correct"} and cell["correct"] <= cell["parsed"]
        for name, vectors in outcomes.items():
            vectors[ident] = flags(rows, name)
        hashes[ident] = digest(file)
        grammar_misses = {}
        for mode in MARKERS:
            missed = [row[mode] for row in rows if row[mode]["symbolic"]["correct"] and not row[mode]["strict"]["correct"]]
            grammar_misses[mode] = {
                "strict_wrong_symbolic_correct": len(missed),
                "strict_unparsed": sum(not x["strict"]["parsed"] for x in missed),
                "marker_present": sum(x["marker_present"] for x in missed),
            }
        stats[ident] = {
            "seconds": meta["seconds"],
            "stops": {mode: dict(Counter(row[mode]["stop"] for row in rows)) for mode in PROMPTS},
            "correct": {cell: int(flags.sum()) for cell, flags in outcomes["correct"][ident].items()},
            "parsed": {cell: int(flags.sum()) for cell, flags in outcomes["parsed"][ident].items()},
            "grammar_misses": grammar_misses,
        }
    base = outcomes["correct"][key(*runs[-1])]
    correct = outcomes["correct"]
    layouts = {
        cell: {arm: np.stack([correct[key(seed, selection["selected_recipe"], arm)][cell] for seed in SEEDS])
               for arm in ARMS}
        for cell in base
    }
    summary = {}
    for cell, values in layouts.items():
        summary[cell] = {
            "untouched": float(base[cell].mean()),
            "mean_exact": {arm: float(row.mean()) for arm, row in values.items()},
            "kam_minus_dot": difference(values["kam"], values["dot"]),
        }
    neutral = layouts["neutral:symbolic"]
    effects = {}
    for mode in MARKERS:
        symbolic = layouts[f"{mode}:symbolic"]
        strict = layouts[f"{mode}:strict"]
        paired_changes = {}
        for label, first, second in (("prompt", neutral, symbolic), ("grading", symbolic, strict),
                                     ("combined", neutral, strict)):
            paired_changes[label] = {
                "change_in_kam_minus_dot": gap_change(first, second),
                "by_arm": {arm: difference(second[arm], first[arm]) for arm in ARMS},
            }
        assert np.isclose(sum(paired_changes[label]["change_in_kam_minus_dot"]["mean_delta"]
                              for label in ("prompt", "grading")),
                          paired_changes["combined"]["change_in_kam_minus_dot"]["mean_delta"])
        effects[mode] = paired_changes
    return {"model": selection["model"], "kl_weight": selection["selected_kl_weight"], "n": 512,
            "seeds": SEEDS, "cells": summary, "interventions": effects, "checkpoints": stats,
            "manifest_sha256": digest(manifest_path), "outcome_sha256": hashes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("model", choices=("smollm2", "llama3"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("selection_dir", type=Path)
    parser.add_argument("gold_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, summarize(args.work, args.model, args.manifest, args.selection_dir, args.gold_audit))
