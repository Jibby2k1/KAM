#!/usr/bin/env python3
"""Compare exact sparse and dense implementations of four-link IIRC beam scoring."""

import argparse
import gzip
import json
from pathlib import Path
import platform
from random import Random
from time import perf_counter_ns

import numpy as np

from fit_kam_iirc_anchor_beam import FOLDS, WIDTHS, fit_models, fold, log_prob
from fit_kam_iirc_pairs import digest, read_features
from fit_kam_iirc_sequential import linear

METHODS = ("dense", "unique", "mask")


def pair_tables(n):
    i, j = np.triu_indices(n, 1)
    lookup = np.full((n, n), -1, dtype=np.int32)
    lookup[i, j] = lookup[j, i] = np.arange(len(i), dtype=np.int32)
    neighbors = np.stack([np.delete(np.arange(n), a) for a in range(n)])
    return i, j, lookup, neighbors


def predict(docs, factors, heads, frozen, variant, mode):
    i, j, lookup, neighbors = frozen
    first = heads["first"]
    doc_scores = docs @ (np.asarray(first["coef"]) / first["scale"])
    top = np.argsort(-doc_scores, kind="stable")[:max(WIDTHS)]
    first_log = log_prob(doc_scores)
    edges = lookup[top[:, None], neighbors[top]]
    if mode == "unique":
        subset, locations = np.unique(edges, return_inverse=True)
        locations = locations.reshape(edges.shape)
    elif mode == "mask":
        visited = np.zeros(len(i), dtype=np.bool_)
        visited[edges] = True
        subset = np.flatnonzero(visited)
        map_local = np.empty(len(i), dtype=np.int32)
        map_local[subset] = np.arange(len(subset))
        locations = map_local[edges]
    else:
        subset = np.arange(len(i))
        locations = edges
    a, b = docs[i[subset]], docs[j[subset]]
    cols = 3 if variant == "neutral" else 5
    feature = np.column_stack((a + b, np.minimum(a, b), factors[subset, :cols]))
    values = linear(feature, heads["conditional"][variant])
    probs = values[locations]
    probs -= np.logaddexp.reduce(probs, axis=1, keepdims=True)
    probs += first_log[top, None]
    merged = np.full(len(subset), -np.inf)
    np.logaddexp.at(merged, locations, probs)
    best = subset[int(merged.argmax())]
    return [int(i[best]), int(j[best])], len(subset)


def run(a):
    indexed, _, info = read_features(a.static, 1200)
    folds = [fit_models([r for r in indexed if fold(r["source_id"]) != k]) for k in range(FOLDS)]
    trace = json.loads(gzip.decompress(a.reference.read_bytes()))
    expected = {r["id"]: r for r in trace}
    assert expected.keys() == {r["id"] for r in indexed}
    frozen = {len(r["docs"]): pair_tables(len(r["docs"])) for r in indexed}
    records = [{"id": r["id"], "source_id": r["source_id"], "cases": {}} for r in indexed]
    prepared = [(np.asarray(r["docs"]), np.asarray(r["pairs"]), folds[fold(r["source_id"])],
                 frozen[len(r["docs"])]) for r in indexed]
    for v in ("neutral", "title"):
        for method in METHODS:
            predict(*prepared[0], v, method)
    rng = Random(64992)
    for row, inputs, saved in zip(indexed, prepared, records, strict=True):
        previous = expected[row["id"]]
        assert previous["source_fold"] == fold(row["source_id"]) and previous["source_id"] == row["source_id"]
        for v in ("neutral", "title"):
            observations = {m: [] for m in METHODS}
            touched = {}
            for _ in range(a.repeats):
                for kind in rng.sample(METHODS, k=len(METHODS)):
                    started = perf_counter_ns()
                    chosen, count = predict(*inputs, v, kind)
                    elapsed = (perf_counter_ns() - started) / 1000
                    assert chosen == previous["selected"][f"beam_{v}_4"], (row["id"], v, kind, chosen)
                    assert count == (previous["distinct_pairs_within_beam"]["4"] if kind != "dense"
                                     else previous["all_unordered_pairs"])
                    observations[kind].append(round(elapsed, 6))
                    touched[kind] = count
            saved["cases"][v] = {"row_counts": touched, "runtime_microseconds": observations}
    return {"static_sha256": digest(a.static), "source": info, "replay_sha256": digest(a.reference),
            "repeats": a.repeats, "host": {"platform": platform.system(), "machine": platform.machine(),
                                           "python": platform.python_version(), "numpy": np.__version__},
            "rows": records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("static", "reference", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    assert a.repeats > 0
    data = json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode()
    a.output.write_bytes(gzip.compress(data, mtime=0))


if __name__ == "__main__":
    main()
