#!/usr/bin/env python3
"""Fit IIRC additive and pair scorers and evaluate a locked fit on held-out articles."""

import argparse
from collections import Counter
from hashlib import file_digest
import gzip
import json
from pathlib import Path

import numpy as np

from analyze_kam_iirc_pair import paired
from analyze_kam_shared_serving import unpack

ARMS = ("lead_additive", "link_additive", "pair_neutral", "pair_title")
PENALTY = 0.05
CONTRASTS = (("pair_neutral", "link_additive"), ("pair_title", "pair_neutral"),
             ("link_additive", "lead_additive"), ("pair_neutral", "frozen_sequential"),
             ("link_additive", "frozen_sequential"), ("pair_title", "frozen_sequential"))
FROZEN_KEYS = ("config_sha256", "baseline_sha256", "assets", "device", "torch", "doc_columns", "pair_columns")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_features(path, expected_count):
    shards = unpack(path)
    assert [s["shard"] for s in shards] == list(range(len(shards)))
    frozen = {k: shards[0][k] for k in FROZEN_KEYS}
    corpus = {k: shards[0][k] for k in ("fixture_sha256", "source_sha256")}
    assert frozen["device"] == "NVIDIA B200"
    assert all(all(s[k] == v for k, v in (frozen | corpus).items()) and s["shards"] == len(shards)
               for s in shards)
    rows = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == len({r["id"] for r in rows}) == expected_count
    for s in shards:
        assert s["rows"] == rows[s["shard"]::len(shards)]
    for r in rows:
        n = len(r["docs"])
        assert n > 2 and len(r["gold"]) == len(set(r["gold"])) == 2
        assert all(0 <= j < n for j in r["gold"])
        assert np.asarray(r["docs"]).shape == (n, len(frozen["doc_columns"]))
        assert np.asarray(r["pairs"]).shape == (n * (n - 1) // 2, len(frozen["pair_columns"]))
    return rows, frozen, corpus


def design(rows):
    counts = [len(r["pairs"]) for r in rows]
    starts = np.cumsum([0, *counts[:-1]])
    ids = np.repeat(np.arange(len(rows)), counts)
    targets, lead, link, neutral, title = [], [], [], [], []
    for start, row in zip(starts, rows, strict=True):
        docs = np.asarray(row["docs"], dtype=np.float64)
        pairs = np.asarray(row["pairs"], dtype=np.float64)
        i, j = np.triu_indices(len(docs), 1)
        sums = docs[i] + docs[j]
        joint = np.column_stack((sums, np.minimum(docs[i], docs[j]), pairs[:, :3]))
        lead.append(sums[:, [0, 1, 4]])
        link.append(sums)
        neutral.append(joint)
        title.append(np.column_stack((joint, pairs[:, 3:])))
        a, b = row["gold"]
        picked = np.flatnonzero((i == min(a, b)) & (j == max(a, b)))
        targets.append(start + int(picked.item()))
    return {arm: np.concatenate(x) for arm, x in zip(ARMS, (lead, link, neutral, title), strict=True)}, (
        starts, ids, np.asarray(targets), np.asarray(counts)
    )


def fit_arm(matrix, index):
    from scipy.optimize import minimize

    starts, ids, targets, counts = index
    n = len(starts)
    means = np.add.reduceat(matrix, starts, axis=0) / counts[:, None]
    mean = means.mean(axis=0)
    squared = np.add.reduceat((matrix - mean) ** 2, starts, axis=0) / counts[:, None]
    scale = squared.mean(axis=0) ** 0.5
    scale[scale < 1e-10] = 1
    matrix = (matrix - mean) / scale
    assert np.isfinite(matrix).all()
    gold_sum = matrix[targets].sum(axis=0)
    def objective(weight):
        logits = matrix @ weight
        shift = np.maximum.reduceat(logits, starts)
        exp = np.exp(logits - shift[ids])
        sums = np.add.reduceat(exp, starts)
        loss = (shift + np.log(sums) - logits[targets]).mean() + PENALTY * (weight @ weight) / 2
        gradient = (matrix.T @ (exp / sums[ids]) - gold_sum) / n + PENALTY * weight
        return loss, gradient
    result = minimize(objective, np.zeros(matrix.shape[1]), method="L-BFGS-B", jac=True,
                      options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 300})
    assert result.success and np.max(np.abs(result.jac)) < 2e-6, result.message
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": result.x.tolist(),
            "objective": float(result.fun), "iterations": int(result.nit),
            "gradient_max": float(np.max(np.abs(result.jac)))}


def fit(train):
    rows, frozen, corpus = read_features(train, 1200)
    assert "train" in corpus["source_sha256"]
    matrices, index = design(rows)
    return {"feature_sha256": digest(train), "training": corpus, "frozen": frozen, "questions": len(rows),
            "source_articles": sorted({r["source_id"] for r in rows}), "l2": PENALTY,
            "arms": {k: fit_arm(x, index) for k, x in matrices.items()}}


def summarize(rows):
    if not rows:
        return None
    methods = (*ARMS, "frozen_sequential")
    return {"questions": len(rows), "source_articles": len({r["source_id"] for r in rows}),
            "both_gold": {arm: sum(r["success"][arm] for r in rows) for arm in methods},
            "paired": {a + "_vs_" + b: paired(rows, a, b) for a, b in CONTRASTS},
            "pair_neutral_or_sequential_oracle": sum(r["success"]["pair_neutral"] or
                                                    r["success"]["frozen_sequential"] for r in rows)}


def evaluate(dev, locked_path, reference):
    locked = json.loads(locked_path.read_text())
    rows, frozen, corpus = read_features(dev, 153)
    assert "dev" in corpus["source_sha256"]
    assert locked["frozen"] == frozen
    assert locked["training"]["source_sha256"]["articles"] == corpus["source_sha256"]["articles"]
    assert set(locked["source_articles"]).isdisjoint(r["source_id"] for r in rows)
    archives = unpack(reference)
    for replay in archives:
        assert all(replay[k] == frozen[k] for k in FROZEN_KEYS if k in replay)
        assert replay["source_sha256"] == corpus["source_sha256"]
    expected = {r["id"]: r for s in archives for r in s["rows"]}
    assert expected.keys() == {r["id"] for r in rows}
    matrices, (starts, _, _, counts) = design(rows)
    predictions = {}
    for arm, matrix in matrices.items():
        model = locked["arms"][arm]
        logits = ((matrix - model["mean"]) / model["scale"]) @ model["coef"]
        predictions[arm] = [int(logits[start:start + n].argmax()) for start, n in zip(starts, counts, strict=True)]
    results = []
    for i, row in enumerate(rows):
        old = expected[row["id"]]
        assert all(old[k] == row[k] for k in ("pool_sha256", "source_id", "gold"))
        assert old["candidates"] == len(row["docs"])
        first, second = np.triu_indices(len(row["docs"]), 1)
        hits = {}
        for arm in ARMS:
            j = predictions[arm][i]
            hits[arm] = [int(first[j]), int(second[j])]
        hits["frozen_sequential"] = old["arms"]["sequential"]["selected"]
        results.append({"id": row["id"], "source_id": row["source_id"], "gold": row["gold"],
                        "gold_title_link": old["gold_title_link"], "selected": hits,
                        "success": {k: set(v) == set(row["gold"]) for k, v in hits.items()}})
    groups = {"all": results, "title_link": [r for r in results if r["gold_title_link"]],
              "no_title_link": [r for r in results if not r["gold_title_link"]]}
    return {"fit_sha256": digest(locked_path), "feature_sha256": digest(dev), "reference_sha256": digest(reference),
            "test": corpus, "pool_widths": dict(sorted(Counter(len(r["docs"]) for r in rows).items())),
            "groups": {k: summarize(v) for k, v in groups.items()}, "rows": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for command, fields in (("fit", ("train", "output")), ("evaluate", ("dev", "lock", "reference", "output"))):
        child = commands.add_parser(command)
        for name in fields:
            child.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = fit(args.train) if args.mode == "fit" else evaluate(args.dev, args.lock, args.reference)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if args.output.suffix == ".gz":
        payload = gzip.compress(payload, mtime=0)
    args.output.write_bytes(payload)


if __name__ == "__main__":
    main()
