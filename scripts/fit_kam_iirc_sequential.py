#!/usr/bin/env python3
"""Compare joint and greedy decoding under matched IIRC supervision."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from analyze_kam_iirc_pair import paired
from fit_kam_iirc_pairs import PENALTY, design, digest, fit_arm, read_features

VARIANTS = ("neutral", "title")
METHODS = ("link_additive", "pair_neutral", "restricted_neutral", "conditional_neutral",
           "pair_title", "restricted_title", "conditional_title")
CONTRASTS = (("conditional_neutral", "pair_neutral"), ("restricted_neutral", "pair_neutral"),
             ("conditional_neutral", "restricted_neutral"), ("conditional_title", "pair_title"),
             ("restricted_title", "pair_title"), ("conditional_title", "restricted_title"))


def orders(row):
    return np.triu_indices(len(row["docs"]), 1)


def branch(i, j, first):
    return np.flatnonzero((i == first) | (j == first))


def fit(train_path, joint_path):
    joint_fit = json.loads(joint_path.read_text())
    train, frozen, metadata = read_features(train_path, 1200)
    assert joint_fit["feature_sha256"] == digest(train_path)
    assert joint_fit["frozen"] == frozen and joint_fit["training"] == metadata
    matrices, (starts, _, targets, _) = design(train)
    rows, gold, sizes = [], [], []
    offset = 0
    for row, start, target in zip(train, starts, targets, strict=True):
        i, j = orders(row)
        for anchor in row["gold"]:
            indices = start + branch(i, j, anchor)
            where = np.flatnonzero(indices == target)
            assert where.size == 1
            rows.append(indices)
            gold.append(offset + where.item())
            sizes.append(len(indices))
            offset += len(indices)
    assert len(rows) == 2 * len(train)
    pick = np.concatenate(rows)
    counts = np.asarray(sizes)
    index = np.cumsum([0, *sizes[:-1]]), np.repeat(np.arange(len(rows)), counts), np.asarray(gold), counts
    return {"joint_fit_sha256": digest(joint_path), "train_sha256": digest(train_path), "frozen": frozen,
            "source_articles": len(set(r["source_id"] for r in train)), "training_questions": len(train),
            "teacher_forced_anchors": len(rows), "l2": PENALTY,
            "models": {k: fit_arm(matrices[f"pair_{k}"][pick], index) for k in VARIANTS}}


def linear(matrix, model):
    return ((matrix - model["mean"]) / model["scale"]) @ model["coef"]


def summarize(rows):
    n = len(rows)
    first_gold = sum(r["first_gold"] for r in rows)
    return {"questions": n, "source_articles": len(set(r["source_id"] for r in rows)),
            "first_gold": first_gold, "first_miss_recovered_by_joint": {
                k: sum(not r["first_gold"] and r["success"][f"pair_{k}"] for r in rows) for k in VARIANTS},
            "both_gold": {m: sum(r["success"][m] for r in rows) for m in METHODS},
            "conditional_oracle_first": {k: sum(r[f"oracle_{k}"] for r in rows) for k in VARIANTS},
            "conditional_second_given_first_gold": {
                k: {"hits": sum(r["success"][f"conditional_{k}"] for r in rows), "first_gold": first_gold}
                for k in VARIANTS},
            "paired": {f"{a}_vs_{b}": paired(rows, a, b) for a, b in CONTRASTS} if n else {}}


def evaluate(dev_path, head_path, joint_path, reference_path):
    heads, joint, ref = (json.loads(p.read_text()) for p in (head_path, joint_path, reference_path))
    assert heads["joint_fit_sha256"] == ref["fit_sha256"] == digest(joint_path)
    assert heads["train_sha256"] == joint["feature_sha256"] and ref["feature_sha256"] == digest(dev_path)
    rows, frozen, metadata = read_features(dev_path, 153)
    assert metadata == ref["test"] and frozen == joint["frozen"] == heads["frozen"]
    assert set(joint["source_articles"]).isdisjoint(r["source_id"] for r in rows)
    reference = {r["id"]: r for r in ref["rows"]}
    assert len(rows) == len(reference) and {r["id"] for r in rows} == reference.keys()
    matrices, (starts, _, _, counts) = design(rows)
    coefs = joint["arms"]["link_additive"]
    first_w = np.asarray(coefs["coef"]) / coefs["scale"]
    scoring = {k: (linear(matrices[f"pair_{k}"], joint["arms"][f"pair_{k}"]),
                   linear(matrices[f"pair_{k}"], heads["models"][k])) for k in VARIANTS}
    results = []
    for row, start, count in zip(rows, starts, counts, strict=True):
        old = reference[row["id"]]
        assert row["source_id"] == old["source_id"] and row["gold"] == old["gold"]
        i, j = orders(row)
        doc_score = np.asarray(row["docs"]) @ first_w
        first = int(doc_score.argmax())
        gold = set(row["gold"])
        selected = {"link_additive": old["selected"]["link_additive"]}
        assert set(np.argsort(-doc_score, kind="stable")[:2]) == set(selected["link_additive"])
        result = {"id": row["id"], "source_id": row["source_id"], "gold": row["gold"],
                  "gold_title_link": old["gold_title_link"], "first": first, "first_gold": first in gold,
                  "selected": selected}
        choices = branch(i, j, first)
        for variant, (all_scores, all_cond) in scoring.items():
            scores, conditioned = (x[start:start + count] for x in (all_scores, all_cond))
            a = int(scores.argmax())
            pair = [int(i[a]), int(j[a])]
            assert pair == old["selected"][f"pair_{variant}"]
            selected[f"pair_{variant}"] = pair
            for method, vector in (("restricted", scores), ("conditional", conditioned)):
                index = choices[int(vector[choices].argmax())]
                second = int(j[index] if i[index] == first else i[index])
                selected[f"{method}_{variant}"] = [first, second]
            successful = []
            for anchor in row["gold"]:
                options = branch(i, j, anchor)
                index = options[int(conditioned[options].argmax())]
                successful.append({i[index], j[index]} == gold)
            result[f"oracle_{variant}"] = any(successful)
        result["success"] = {method: set(pair) == gold for method, pair in selected.items()}
        results.append(result)
    groups = {"all": results, "title_link": [r for r in results if r["gold_title_link"]],
              "no_title_link": [r for r in results if not r["gold_title_link"]]}
    return {"dev_sha256": digest(dev_path), "head_sha256": digest(head_path),
            "joint_fit_sha256": digest(joint_path), "reference_sha256": digest(reference_path),
            "by_candidate_count": dict(sorted(Counter(len(r["docs"]) for r in rows).items())),
            "cohorts": {k: summarize(v) for k, v in groups.items()}, "rows": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="mode", required=True)
    for mode, args in (("fit", ("train", "joint-fit", "output")), ("evaluate", ("dev", "conditional-fit", "joint-fit", "reference", "output"))):
        command = commands.add_parser(mode)
        for name in args:
            command.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = (fit(a.train, a.joint_fit) if a.mode == "fit" else
              evaluate(a.dev, a.conditional_fit, a.joint_fit, a.reference))
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
