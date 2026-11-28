#!/usr/bin/env python3
"""Verify checkpoint lineage and summarize the marker-free MATH-500 comparison."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, LLAMA_ID, MODEL_ID, SEEDS, digest, key
from kam_math_answer_type_replication import PROTOCOL as REPLICATION_PROTOCOL
from kam_math_kl_protocol import BATCH_SIZE, SOURCES, test_runs, write_json
from summarize_kam_adapter_preservation import difference, read_archive

MODES = ("full", "prefix", "terminal")
PROMPT_SUFFIX = "\n\nSolve the problem. Explain your reasoning and state your final answer clearly."


def truth(rows, mode, name="correct"):
    return np.asarray([row[mode][name] for row in rows], dtype=bool)


def split_metrics(mask, flags, baseline):
    result = {"n": int(mask.sum()), "untouched": {}, "mean_exact": {}, "contrasts": {}}
    for mode in MODES:
        base = baseline[mode][None, mask]
        result["untouched"][mode] = float(base.mean())
        arms = {arm: flags[arm][mode][:, mask] for arm in ARMS}
        result["mean_exact"][mode] = {arm: float(results.mean()) for arm, results in arms.items()}
        result["contrasts"][mode] = difference(arms["kam"], arms["dot"])
        if mask.all():
            result["contrasts"][f"{mode}:kam-minus-untouched"] = difference(arms["kam"], base)
            result["contrasts"][f"{mode}:dot-minus-untouched"] = difference(arms["dot"], base)
    return result


def summarize(work, model, old_dir, gold_audit, replication_manifest=None):
    selection = old_dir / f"{model}-selection.json"
    selected = json.loads(selection.read_text())
    recipe = selected["selected_recipe"]
    runs = test_runs(recipe)
    old = read_archive(old_dir / f"{model}-outcomes.json.gz")
    replication = None if replication_manifest is None else json.loads(replication_manifest.read_text())
    if replication is None:
        fixture = json.loads((old_dir / "fixture-manifest.json").read_text())
        integer_set = set(fixture["fixtures"]["test"]["ids"])
        source_sha = SOURCES["math500-test.jsonl"]
        expected_protocol, expected_size = "kam-math500-neutral-v1", 500
    else:
        assert replication["protocol"] == REPLICATION_PROTOCOL
        pairs = replication["pairs"]
        integer_set = {pair["integer_id"] for pair in pairs}
        source_sha = replication["fixture_sha256"]
        expected_protocol, expected_size = REPLICATION_PROTOCOL, replication["n"]
        assert expected_size == 2 * len(integer_set) == 2 * len(pairs)
    grader = json.loads(gold_audit.read_text())
    expected_model = MODEL_ID if model == "smollm2" else LLAMA_ID
    payloads = {}
    hashes = {}
    signatures = None
    for run in runs:
        ident = key(*run)
        path = work / f"graded-{ident}.json"
        payload = json.loads(path.read_text())
        assert set(payload) == {"metadata", "records"}
        meta, rows = payload["metadata"], payload["records"]
        original = old(f"test-{ident}")["metadata"]
        assert meta["protocol"] == expected_protocol
        if replication is not None:
            assert meta["manifest_sha256"] == digest(replication_manifest)
            assert [row["id"] for row in rows] == [ident for pair in pairs for ident in (
                pair["integer_id"], pair["noninteger_id"],
            )]
        assert meta["model"] == expected_model and (meta["seed"], meta["recipe"], meta["arm"]) == run
        assert meta["batch_size"] == BATCH_SIZE and (meta["prefix_budget"], meta["budget"]) == (384, 1024)
        assert meta["prompt_suffix"] == PROMPT_SUFFIX
        assert meta["source_sha256"] == source_sha == grader["source_sha256"]
        assert meta["selection_sha256"] == original["selection_sha256"] == digest(selection)
        assert meta["grader_protocol"] == grader["protocol"] and meta["grader_versions"] == grader["versions"]
        for field in ("revision", "adapter_sha256", "model_fingerprints", "new_trainable_parameters", "training", "gpu"):
            assert meta[field] == original[field], (ident, field)
        assert len(rows) == len({r["id"] for r in rows}) == grader["self_verifying_answers"] == expected_size
        signature = [(r["id"], r["subject"], r["level"], r["prompt_tokens"]) for r in rows]
        signatures = signature if signatures is None else signatures
        assert signature == signatures, ident
        for r in rows:
            assert set(r) == {"id", "subject", "level", "prompt_tokens", "output_tokens", "stop", *MODES}
            assert (r["stop"], r["output_tokens"] == 1024) in (("budget", True), ("eos", False))
            assert 0 <= r["output_tokens"] <= 1024 and r["prompt_tokens"] > 0
            assert r["output_tokens"] > 384 or r["full"] == r["prefix"]
            for mode in MODES:
                assert set(r[mode]) == {"correct", "parsed"} and r[mode]["correct"] <= r[mode]["parsed"]
        hashes[ident] = digest(path)
        payloads[ident] = payload

    base_id = key(*runs[-1])
    baseline = {mode: truth(payloads[base_id]["records"], mode) for mode in MODES}
    by_arm = {arm: [payloads[key(seed, recipe, arm)]["records"] for seed in SEEDS] for arm in ARMS}
    flags = {arm: {mode: np.stack([truth(rows, mode) for rows in arm_rows]) for mode in MODES}
             for arm, arm_rows in by_arm.items()}
    subjects = np.array([subject for _, subject, _, _ in signatures])
    levels = np.array([level for _, _, level, _ in signatures])
    integer = np.array([ident in integer_set for ident, _, _, _ in signatures])
    assert int(integer.sum()) == len(integer_set)
    groups = {"full": np.ones(expected_size, dtype=bool), "integer": integer, "noninteger": ~integer}
    groups.update({f"subject:{subject}": subjects == subject for subject in sorted(set(subjects))})
    groups.update({f"level:{level}": levels == level for level in sorted(set(levels))})
    checkpoints = {}
    for ident, payload in payloads.items():
        rows = payload["records"]
        full = truth(rows, "full")
        prefix = truth(rows, "prefix")
        checkpoints[ident] = {
            "correct": {mode: int(truth(rows, mode).sum()) for mode in MODES},
            "parsed": {mode: int(truth(rows, mode, "parsed").sum()) for mode in MODES},
            "full_vs_prefix": {"gained": int((full & ~prefix).sum()), "lost": int((prefix & ~full).sum())},
            "full_vs_base": {"gained": int((full & ~baseline["full"]).sum()),
                             "lost": int((baseline["full"] & ~full).sum())},
            "stops": dict(Counter(row["stop"] for row in rows)),
            "credited_at_cap": sum(row["stop"] == "budget" and row["full"]["correct"] for row in rows),
            "terminal_changed": int((full != truth(rows, "terminal")).sum()),
            "grader_diagnostics": dict(Counter(f"{x['mode']}:{x['error']}" for x in payload["metadata"]["grader_diagnostics"])),
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "seconds": payload["metadata"]["seconds"],
        }
    eos_only = {
        arm: np.stack([truth(rows, "full") & np.array([row["stop"] == "eos" for row in rows])
                       for rows in arm_rows])
        for arm, arm_rows in by_arm.items()
    }
    result = {"model": expected_model, "recipe": recipe, "kl_weight": selected["selected_kl_weight"],
            "groups": {label: split_metrics(mask, flags, baseline) for label, mask in groups.items()},
            "checkpoints": checkpoints, "seeds": SEEDS, "outcome_sha256": hashes,
            "cap_zero_sensitivity": {"mean_exact": {arm: float(rows.mean()) for arm, rows in eos_only.items()},
                                     "kam_minus_dot": difference(eos_only["kam"], eos_only["dot"])},
            "source_sha256": grader["source_sha256"], "selection_sha256": digest(selection)}
    if replication is not None:
        for index, pair in enumerate(pairs):
            assert all(signature[1:3] == (pair["subject"], pair["level"])
                       for signature in signatures[index * 2:index * 2 + 2])
        result["manifest_sha256"] = digest(replication_manifest)
        result["paired_answer_type_interaction"] = {}
        for mode in MODES:
            kam, dot = flags["kam"][mode], flags["dot"][mode]
            result["paired_answer_type_interaction"][mode] = difference(
                kam[:, 1::2].astype(np.int8) + dot[:, ::2],
                dot[:, 1::2].astype(np.int8) + kam[:, ::2],
            )
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("model", choices=("smollm2", "llama3"))
    cli.add_argument("old_results", type=Path)
    cli.add_argument("gold_audit", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--replication-manifest", type=Path)
    args = cli.parse_args()
    write_json(args.output, summarize(args.work, args.model, args.old_results, args.gold_audit,
                                      args.replication_manifest))
