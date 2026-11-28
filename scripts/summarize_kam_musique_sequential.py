#!/usr/bin/env python3
"""Combine frozen first-hop choices with the matching endpoint logits."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_kam_pretrained_endpoint_selection import BOTH_MODES, BOTH_PROTOCOL, EARLIER_PROTOCOL, PROTOCOL
from kam_adapter_preservation import ARMS, SEEDS, digest, key
from kam_math_kl_protocol import test_runs, write_json
from summarize_kam_adapter_preservation import read_archive
from summarize_kam_pretrained_endpoint_selection import family_difference


def fields(rows, mode, field):
    return np.asarray([row[mode][field] for row in rows]).reshape((-1, 2))


def observations(single, both, earlier):
    expected = np.array([row["expected"] for row in single]).reshape((-1, 2))
    assert np.all(expected[:, 0] + expected[:, 1] == 1)
    # First-hop columns are indexed by the correct passage's physical slot.
    physical_order = expected.argsort(axis=1)
    first_correct = np.take_along_axis(fields(earlier, "earlier_choice", "picked") == expected, physical_order, axis=1)
    first_tokens = fields(earlier, "earlier_choice", "tokens")
    assert np.all(first_tokens[:, 0] == first_tokens[:, 1])
    true = fields(single, "true_prefix", "picked") == expected
    cross = fields(single, "cross_prefix", "picked") == expected
    sequential = np.where(first_correct[:, :, None], true[:, None, :], cross[:, None, :])
    direct = np.stack([fields(both, mode, "picked") == expected for mode in BOTH_MODES], axis=1)
    direct_tokens = np.stack([fields(both, mode, "tokens") for mode in BOTH_MODES], axis=1)
    true_tokens = fields(single, "true_prefix", "tokens")
    assert np.array_equal(true_tokens, fields(single, "cross_prefix", "tokens"))
    collect = {
        "direct_all_four": direct.all(axis=(1, 2)),
        "sequential_all_four": sequential.all(axis=(1, 2)),
        "first_hop_both_orders": first_correct.all(axis=1),
        "gold_first_endpoint_reference": true.all(axis=1),
        "endpoint_path_oracle": true.all(axis=1) | cross.all(axis=1),
        "direct_earlier_first": direct[:, 0].all(axis=1),
        "direct_earlier_second": direct[:, 1].all(axis=1),
        "sequential_earlier_first": sequential[:, 0].all(axis=1),
        "sequential_earlier_second": sequential[:, 1].all(axis=1),
    }
    first_broadcast = np.broadcast_to(first_correct[:, :, None], sequential.shape)
    mechanics = {
        "first_hop_correct_by_position": first_correct.sum(axis=0).tolist(),
        "first_hop_correct": int(first_broadcast.sum()),
        "first_correct_endpoint_correct": int((first_broadcast & sequential).sum()),
        "first_incorrect_endpoint_correct": int((~first_broadcast & sequential).sum()),
        "presentation_count": int(sequential.size),
        "mean_direct_prompt_tokens": float(direct_tokens.mean()),
        "mean_sequential_prompt_tokens": float((first_tokens[:, 0, None, None] + true_tokens[:, None, :]).mean()),
    }
    return collect, mechanics


def summarize(first_dir, results_dir, selection_path):
    manifest_path = results_dir / "earlier-choice-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"] == EARLIER_PROTOCOL
    original = read_archive(results_dir / "llama3-outcomes.json.gz")
    both = read_archive(results_dir / "llama3-both-earlier-outcomes.json.gz")
    selection = json.loads(selection_path.read_text())
    recipe = selection["selected_recipe"]
    assert selection["selected_kl_weight"] == .5
    signatures = [(c["pair"], c["family"], q, reverse, i ^ int(reverse))
                  for c in manifest["cases"] for i, q in enumerate(c["question_ids"])
                  for reverse in (False, True)]
    families = [s[1] for s in signatures[::2]]
    counts, checkpoints, hashes = {}, {}, {}
    runs = test_runs(recipe)
    for run in runs:
        name = key(*run)
        first_path = first_dir / f"endpoint-{name}.json"
        first = json.loads(first_path.read_text())
        prefix, direct = original(first_path.stem), both(first_path.stem)
        for doc, protocol, manifest_sha in (
            (first, EARLIER_PROTOCOL, digest(manifest_path)),
            (prefix, PROTOCOL, manifest["single_earlier_manifest_sha256"]),
            (direct, BOTH_PROTOCOL, manifest["both_earlier_manifest_sha256"]),
        ):
            meta = doc["metadata"]
            assert (meta["seed"], meta["recipe"], meta["arm"]) == run
            assert (meta["protocol"], meta["manifest_sha256"]) == (protocol, manifest_sha)
            assert meta["selection_sha256"] == digest(selection_path)
            for field in ("model", "revision", "gpu", "model_fingerprints", "adapter_sha256", "training"):
                assert meta[field] == first["metadata"][field]
            assert [tuple(r[k] for k in ("pair", "family", "question", "reverse", "expected"))
                    for r in doc["records"]] == signatures
        for row in first["records"]:
            choice = row["earlier_choice"]
            assert np.isfinite(choice["b_minus_a"]) and int(choice["b_minus_a"] > 0) == choice["picked"]
        values, info = observations(prefix["records"], direct["records"], first["records"])
        checkpoints[name] = values
        counts[name] = {**info, "questions_correct": {mode: int(flag.sum()) for mode, flag in values.items()}}
        hashes[name] = digest(first_path)
    baseline = checkpoints[key(*runs[-1])]
    groups = {mode: {arm: np.stack([checkpoints[key(seed, recipe, arm)][mode] for seed in SEEDS]) for arm in ARMS}
              for mode in baseline}
    summaries = {mode: {"untouched": float(baseline[mode].mean()),
                        "means": {arm: float(value.mean()) for arm, value in fits.items()},
                        "kam_minus_dot": family_difference(fits["kam"], fits["dot"], families)}
                 for mode, fits in groups.items()}
    seq, direct = groups["sequential_all_four"], groups["direct_all_four"]
    comparison = {"changes": {arm: family_difference(seq[arm], direct[arm], families) for arm in ARMS},
                  "change_in_kam_minus_dot": family_difference(
                      seq["kam"].astype(np.int8) + direct["dot"], seq["dot"].astype(np.int8) + direct["kam"], families
                  )}
    return {"n": len(families), "families": len(set(families)), "seeds": SEEDS, "results": summaries,
            "comparison": comparison, "checkpoints": counts, "outcome_sha256": hashes,
            "manifest_sha256": digest(manifest_path)}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("first_dir", "results_dir", "selection", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    write_json(args.output, summarize(args.first_dir, args.results_dir, args.selection))
