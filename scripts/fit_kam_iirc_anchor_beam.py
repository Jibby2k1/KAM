#!/usr/bin/env python3
"""Cross-fit IIRC joint and anchor-beam policies on disjoint source articles."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

import numpy as np

from analyze_kam_iirc_pair import paired
from fit_kam_iirc_pairs import PENALTY, design, digest, fit_arm, read_features
from fit_kam_iirc_sequential import VARIANTS, branch, linear, orders

FOLDS = 5
SALT = "kam-iirc-anchor-beam-v1:"
WIDTHS = (1, 2, 4)


def fold(source):
    key = int.from_bytes(sha256((SALT + source).encode()).digest()[:8], "big")
    return key % FOLDS


def fit_models(rows):
    matrices, index = design(rows)
    starts, _, targets, _ = index
    branches, sizes, gold = [], [], []
    offset = 0
    for row, start, target in zip(rows, starts, targets, strict=True):
        i, j = orders(row)
        for anchor in row["gold"]:
            ids = start + branch(i, j, anchor)
            branches.append(ids)
            sizes.append(len(ids))
            gold.append(offset + np.flatnonzero(ids == target).item())
            offset += len(ids)
    restricted = np.concatenate(branches)
    meta = np.cumsum([0, *sizes[:-1]]), np.repeat(np.arange(len(sizes)), sizes), np.asarray(gold), np.asarray(sizes)
    return {"first": fit_arm(matrices["link_additive"], index),
            "joint": {v: fit_arm(matrices[f"pair_{v}"], index) for v in VARIANTS},
            "conditional": {v: fit_arm(matrices[f"pair_{v}"][restricted], meta) for v in VARIANTS}}


def log_prob(scores):
    return scores - np.logaddexp.reduce(scores)


def predict(rows, heads):
    matrices, (starts, _, _, counts) = design(rows)
    first = heads["first"]
    first_coef = np.asarray(first["coef"]) / first["scale"]
    score = {v: (linear(matrices[f"pair_{v}"], heads["joint"][v]),
                 linear(matrices[f"pair_{v}"], heads["conditional"][v])) for v in VARIANTS}
    records = []
    for row, start, count in zip(rows, starts, counts, strict=True):
        i, j = orders(row)
        gold = set(row["gold"])
        doc_scores = np.asarray(row["docs"]) @ first_coef
        top = np.argsort(-doc_scores, kind="stable")[:max(WIDTHS)]
        top = [int(x) for x in top]
        first_log = log_prob(doc_scores)
        selected = {"additive": sorted(top[:2])}
        oracle = {}
        for v, (all_scores, all_cond) in score.items():
            joint, cond = (x[start:start+count] for x in (all_scores, all_cond))
            joint_best = joint.argmax()
            selected[f"joint_{v}"] = [int(i[joint_best]), int(j[joint_best])]
            merged = np.full(count, -np.inf)
            oracle[v] = {}
            possible = False
            for depth, anchor in enumerate(top, start=1):
                choices = branch(i, j, anchor)
                probs = log_prob(cond[choices]) + first_log[anchor]
                branch_best = choices[probs.argmax()]
                possible |= {i[branch_best], j[branch_best]} == gold
                merged[choices] = np.logaddexp(merged[choices], probs)
                for width in WIDTHS:
                    if depth == min(width, len(top)):
                        winner = merged.argmax()
                        selected[f"beam_{v}_{width}"] = [int(i[winner]), int(j[winner])]
                        oracle[v][str(width)] = possible
        n = len(row["docs"])
        inspected = {str(d): sum(n - 1 - k for k in range(min(d, n))) for d in WIDTHS}
        records.append({"id": row["id"], "source_id": row["source_id"], "gold": row["gold"], "first_ids": top,
                        "all_unordered_pairs": count.item(), "distinct_pairs_within_beam": inspected,
                        "anchor_contains_gold": {str(d): any(x in gold for x in top[:d]) for d in WIDTHS},
                        "selected": selected, "oracle": oracle,
                        "success": {k: set(v) == gold for k, v in selected.items()}})
    return records


def summarize(rows):
    keys = rows[0]["success"]
    return {"questions": len(rows), "sources": len({r["source_id"] for r in rows}),
            "both_gold": {m: sum(r["success"][m] for r in rows) for m in keys},
            "first_beam_has_gold": {str(d): sum(r["anchor_contains_gold"][str(d)] for r in rows) for d in WIDTHS},
            "branch_oracle": {v: {str(d): sum(r["oracle"][v][str(d)] for r in rows)
                                  for d in WIDTHS} for v in VARIANTS},
            "paired": ({f"beam_{v}_{d}_vs_1": paired(rows, f"beam_{v}_{d}", f"beam_{v}_1")
                        for v in VARIANTS for d in WIDTHS[1:]} |
                       {f"beam_{v}_{d}_vs_joint": paired(rows, f"beam_{v}_{d}", f"joint_{v}")
                        for v in VARIANTS for d in WIDTHS}),
            "mean_distinct_pairs": {"joint": round(mean(r["all_unordered_pairs"] for r in rows), 6)} |
                                   {str(d): round(mean(r["distinct_pairs_within_beam"][str(d)] for r in rows), 6)
                                    for d in WIDTHS}}


def evaluate(static_path):
    rows, _, corpus = read_features(static_path, 1200)
    results, folds = [], []
    for k in range(FOLDS):
        test = [r for r in rows if fold(r["source_id"]) == k]
        train = [r for r in rows if fold(r["source_id"]) != k]
        assert {r["source_id"] for r in train}.isdisjoint(r["source_id"] for r in test)
        report = predict(test, fit_models(train))
        for r in report:
            r["source_fold"] = k
        results.extend(report)
        folds.append({"fold": k, "fit_questions": len(train), **summarize(report)})
    results.sort(key=lambda r: r["id"])
    assert [r["id"] for r in results] == [r["id"] for r in rows]
    return {"feature_sha256": digest(static_path), "fixture": corpus, "fold_salt": SALT,
            "l2": PENALTY, "overall": summarize(results), "folds": folds, "rows": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--static", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    a = p.parse_args()
    result = evaluate(a.static)
    rows = result.pop("rows")
    data = json.dumps(rows, separators=(",", ":"), allow_nan=False).encode()
    a.replay.write_bytes(gzip.compress(data, mtime=0))
    result["replay_payload_sha256"] = sha256(data).hexdigest()
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
