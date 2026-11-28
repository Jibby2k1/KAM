#!/usr/bin/env python3
"""Verify paired training schedules and summarize independent transfer."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from kam_adapter_preservation import digest
from kam_math_kl_protocol import write_json

METRICS = ("nll", "retrieval", "linked", "unlinked", "within_train_length", "beyond_train_length")


def estimate(delta):
    n = delta.shape[1]
    draws = np.random.default_rng(465761).multinomial(n, np.full(n, 1/n), size=20_000)
    sampled = draws @ delta.mean(axis=0) / n
    return {"mean_delta": float(delta.mean()), "by_seed": delta.mean(axis=1).tolist(),
            "ci95_over_eval_units_given_seeds": np.quantile(sampled, (.025, .975)).tolist()}


def read_eval(path, manifest_path, manifest, gpu):
    spec = json.loads(path.read_text())
    assert spec["input_manifest_sha256"] == digest(manifest_path) and spec["gpu"] == gpu
    assert (spec["model"], spec["revision"]) == (manifest["model"], manifest["revision"])
    results = {}
    order = [(r["id"], r["cohort"], r["expected"]) for r in manifest["retrieval"]]
    assert all(order[i][0:2] == order[i+1][0:2] and (order[i][2], order[i+1][2]) == (0, 1) for i in range(0, len(order), 2))
    cohorts = np.array([row[1] for row in order[::2]])
    lengths = np.array([r["tokens"] for r in manifest["retrieval"]]).reshape(-1, 2).max(axis=1)
    subsets = {cohort: cohorts == cohort for cohort in ("linked", "unlinked")}
    subsets.update(within_train_length=lengths <= 512, beyond_train_length=lengths > 512)
    for step, data in spec["checkpoints"].items():
        books = data["pg19"]
        assert [r["id"] for r in books] == [r["id"] for r in manifest["pg19"]]
        p = np.array([row["nll"] for row in books])
        assert p.shape == (24, 8) and np.isfinite(p).all() and (p > 0).all()
        rows = data["retrieval"]
        assert [(r["id"], r["cohort"], r["expected"]) for r in rows] == order
        for row in rows:
            assert np.isfinite(row["margin"]) and row["correct"] == (row["margin"] != 0 and int(row["margin"] > 0) == row["expected"])
        correct = np.array([r["correct"] for r in rows]).reshape(-1, 2).all(axis=1)
        results[int(step)] = {"nll": p.mean(axis=1), "retrieval": correct.astype(float),
                              **{label: correct[mask].astype(float) for label, mask in subsets.items()}}
    return spec, results


def summarize(work, protocol_path, manifest_path):
    protocol, manifest = (json.loads(p.read_text()) for p in (protocol_path, manifest_path))
    seeds, arms, steps = (protocol[k] for k in ("seeds", "arms", "checkpoints"))
    reference_file = work / "eval-0-pretrained.json"
    meta = json.loads(reference_file.read_text())
    _, baseline = read_eval(reference_file, manifest_path, manifest, meta["gpu"])
    assert set(baseline) == {0} and meta["checkpoints"]["0"]["adapter_sha256"] is None
    metrics = {step: {} for step in steps}
    timing = {step: {} for step in steps}
    train_hashes, eval_hashes = {}, {reference_file.stem: digest(reference_file)}
    identity = {}
    for seed in seeds:
        paired = None
        for arm in arms:
            name = f"{seed}-{arm}"
            fit_path, eval_path = (work / f"{mode}-{name}.json" for mode in ("train", "eval"))
            fit = json.loads(fit_path.read_text())
            info = fit["identity"]
            identity[name] = info
            common = {k: v for k, v in info.items() if k != "arm"}
            if paired is None:
                paired = common
            assert common == paired and (info["seed"], info["arm"]) == (seed, arm)
            assert info["protocol_sha256"] == digest(protocol_path) and info["gpu"] == meta["gpu"]
            assert info["tokens_per_step"] == protocol["context"] * sum(protocol["microbatch"])
            assert (info["model"], info["revision"]) == (manifest["model"], manifest["revision"])
            scored, observations = read_eval(eval_path, manifest_path, manifest, meta["gpu"])
            assert set(observations) == set(steps) and [r["step"] for r in fit["history"]] == steps
            last = 0.
            for row in fit["history"]:
                step = row["step"]
                assert row["adapter_sha256"] == scored["checkpoints"][str(step)]["adapter_sha256"]
                assert np.isfinite(row["train_seconds"]) and row["train_seconds"] > last
                last = row["train_seconds"]
                metrics[step][seed, arm] = observations[step]
                timing[step][seed, arm] = row["train_seconds"]
            train_hashes[fit_path.stem], eval_hashes[eval_path.stem] = digest(fit_path), digest(eval_path)

    summary = {}
    for step in steps:
        result = {}
        for metric in METRICS:
            grouped = {arm: np.stack([metrics[step][seed, arm][metric] for seed in seeds]) for arm in arms}
            sl, ss, kl, ks = (grouped[a] for a in arms)
            result[metric] = {"n": len(baseline[0][metric]), "untouched": float(baseline[0][metric].mean()),
                              "arms": {arm: {"mean": float(a.mean()), "by_seed": a.mean(axis=1).tolist(),
                                              "vs_soft_linear": estimate(a-sl)} for arm, a in grouped.items()},
                              "silu_effect": estimate(((ss-sl)+(ks-kl))*.5), "top4_effect": estimate(((kl-sl)+(ks-ss))*.5),
                              "interaction": estimate((ks-kl)-(ss-sl))}
            if metric == "nll":
                result[metric]["untouched_ppl"] = math.exp(result[metric]["untouched"])
                for row in result[metric]["arms"].values():
                    row["ppl"] = math.exp(row["mean"])
        result["train_seconds"] = {arm: [timing[step][seed, arm] for seed in seeds] for arm in arms}
        summary[step] = result
    return {"input_manifest_sha256": digest(manifest_path), "training_sha256": digest(protocol_path), "gpu": meta["gpu"],
            "fits": identity, "train_sha256": train_hashes, "eval_sha256": eval_hashes, "steps": summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("work", "training", "inputs", "output"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    write_json(a.output, summarize(a.work, a.training, a.inputs))
