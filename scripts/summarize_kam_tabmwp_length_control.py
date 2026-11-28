#!/usr/bin/env python3
"""Compare different-topic and same-topic archive tables at matched length."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, digest, key
from kam_math_kl_protocol import test_runs, write_json
from kam_tabmwp_archive_noise import UNRELATED_PROTOCOL
from summarize_kam_adapter_preservation import read_archive
from summarize_kam_pretrained_endpoint_selection import family_difference


def summarize(work, benchmark, selection_path):
    manifest_path = benchmark / "unrelated-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"] == UNRELATED_PROTOCOL
    assert manifest["original_manifest_sha256"] == digest(benchmark / "fixture-manifest.json")
    ids = [c["id"] for c in manifest["cases"]]
    families = [c["family"] for c in manifest["cases"]]
    assert len(ids) == len(set(ids)) == manifest["n"]
    old = read_archive(benchmark / "llama3-outcomes.json.gz")
    recipe = json.loads(selection_path.read_text())["selected_recipe"]
    runs = test_runs(recipe)
    flags, checkpoints, hashes = {}, {}, {}
    for run in runs:
        name = key(*run)
        new_path = work / f"tabmwp-{name}.json"
        new = json.loads(new_path.read_text())
        prior = old(new_path.stem)
        a, b = new["metadata"], prior["metadata"]
        assert (a["seed"], a["recipe"], a["arm"]) == run
        assert a["protocol"] == UNRELATED_PROTOCOL and a["manifest_sha256"] == digest(manifest_path)
        assert a["baseline_generation_sha256"] == b["generation_sha256"]
        assert a["selection_sha256"] == b["selection_sha256"] == digest(selection_path)
        assert a["unmatched_unchanged"] == len(prior["records"]) - manifest["n"]
        for field in ("model", "revision", "gpu", "model_fingerprints", "adapter_sha256", "training", "grader_versions"):
            assert a[field] == b[field]
        assert [r["id"] for r in new["records"]] == ids
        prior_rows = {r["id"]: r for r in prior["records"]}
        for r in new["records"]:
            earlier = prior_rows[r["id"]]
            assert (r["grade"], r["family"]) == (earlier["grade"], earlier["family"])
            assert r["archive_last"]["prompt_tokens"] == earlier["archive_last"]["prompt_tokens"]
        outcomes = {
            "current_only": [prior_rows[i]["current_only"] for i in ids],
            "same_topic": [prior_rows[i]["archive_last"] for i in ids],
            "different_topic": [r["archive_last"] for r in new["records"]],
        }
        flags[name] = {k: np.array([r["correct"] for r in v], dtype=bool) for k, v in outcomes.items()}
        checkpoints[name] = {k: {"correct": int(flags[name][k].sum()), "parsed": sum(r["parsed"] for r in v),
                                  "capped": sum(r["stop"] == "budget" for r in v)} for k, v in outcomes.items()}
        hashes[name] = digest(new_path)
    base = flags[key(*runs[-1])]
    modes = {mode: {arm: np.stack([flags[key(seed, recipe, arm)][mode] for seed in SEEDS]) for arm in ARMS}
             for mode in base}
    summary = {
        mode: {"untouched": float(base[mode].mean()), "means": {arm: float(p.mean()) for arm, p in group.items()},
               "kam_minus_dot": family_difference(group["kam"], group["dot"], families)}
        for mode, group in modes.items()
    }
    after, before = modes["different_topic"], modes["same_topic"]
    contrast = {
        "per_arm": {arm: family_difference(after[arm], before[arm], families) for arm in ARMS},
        "untouched_change": float(base["different_topic"].mean() - base["same_topic"].mean()),
        "change_in_kam_minus_dot": family_difference(
            after["kam"].astype(np.int8) + before["dot"], after["dot"].astype(np.int8) + before["kam"], families,
        ),
    }
    return {"n": len(ids), "families": len(set(families)), "seeds": SEEDS, "accuracy": summary,
            "different_minus_same": contrast, "checkpoints": checkpoints, "outcome_sha256": hashes,
            "manifest_sha256": digest(manifest_path)}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "benchmark", "selection", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    write_json(args.output, summarize(args.work, args.benchmark, args.selection))
