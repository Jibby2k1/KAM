#!/usr/bin/env python3
"""Test dynamic second-hop features while holding the first IIRC link fixed."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_kam_iirc_pair import paired
from analyze_kam_shared_serving import unpack
from fit_kam_iirc_pairs import PENALTY, design, digest, fit_arm, read_features
from fit_kam_iirc_sequential import VARIANTS, branch, linear, orders

ADDED_COLUMNS = ("conditioned_article_cosine", "conditioned_link_excerpt_cosine")


def load(static_path, conditioned_path, reference_path=None):
    n = 153 if reference_path else 1200
    static, fixed, source = read_features(static_path, n)
    shards = unpack(conditioned_path)
    assert [s["shard"] for s in shards] == list(range(len(shards)))
    shared = {k: fixed[k] for k in ("assets", "config_sha256", "baseline_sha256")} | source
    expected = shared | {"first_sha256": digest(reference_path) if reference_path else None}
    assert all(all(s[k] == v for k, v in expected.items()) and s["shards"] == len(shards) for s in shards)
    hardware = {s["gpu"] for s in shards}
    assert hardware == {"NVIDIA L4"}
    feat = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(feat) == len(static)
    for s in shards:
        assert s["rows"] == feat[s["shard"]::len(shards)]
    reference = json.loads(reference_path.read_text()) if reference_path else None
    old = reference["rows"] if reference else [None] * n
    if reference:
        assert reference["dev_sha256"] == digest(static_path)
        assert len(old) == n
    for a, b, prior in zip(static, feat, old, strict=True):
        assert all(a[k] == b[k] for k in ("id", "source_id", "pool_sha256"))
        anchors = set(a["gold"])
        if prior:
            assert prior["id"] == a["id"] and prior["source_id"] == a["source_id"] and prior["gold"] == a["gold"]
            anchors.add(prior["first"])
        assert set(map(int, b["conditioned"])) == anchors
        for v in b["conditioned"].values():
            array = np.asarray(v)
            assert array.shape == (len(a["docs"]), len(ADDED_COLUMNS)) and np.isfinite(array).all()
    return static, feat, old, expected


def conditioned_branch(i, j, anchor, values):
    ids = branch(i, j, anchor)
    partner = np.where(i[ids] == anchor, j[ids], i[ids])
    return ids, np.asarray(values[str(anchor)])[partner]


def fit(static_path, conditioned_path):
    rows, dynamic, _, metadata = load(static_path, conditioned_path)
    matrix, (starts, _, targets, _) = design(rows)
    choices, extras, sizes, gold = [], [], [], []
    offset = 0
    for row, item, start, target in zip(rows, dynamic, starts, targets, strict=True):
        i, j = orders(row)
        for anchor in row["gold"]:
            ids, extra = conditioned_branch(i, j, anchor, item["conditioned"])
            absolute = ids + start
            choices.append(absolute)
            extras.append(extra)
            gold.append(offset + np.flatnonzero(absolute == target).item())
            sizes.append(len(ids))
            offset += len(ids)
    indices, extra = np.concatenate(choices), np.concatenate(extras)
    assert len(sizes) == 2 * len(rows) == 2400
    index = np.cumsum([0, *sizes[:-1]]), np.repeat(np.arange(len(sizes)), sizes), np.asarray(gold), np.asarray(sizes)
    return {"train_static_sha256": digest(static_path), "train_conditioned_sha256": digest(conditioned_path),
            "conditioned_provenance": metadata, "added_columns": ADDED_COLUMNS, "l2": PENALTY,
            "teacher_forced_decisions": len(sizes),
            "models": {k: fit_arm(np.column_stack((matrix[f"pair_{k}"][indices], extra)), index) for k in VARIANTS}}


def summarize(rows):
    return {"questions": len(rows), "source_articles": len({r["source_id"] for r in rows}),
            "both_gold": {a: sum(r["success"][a] for r in rows) for a in
                          ("conditional_neutral", "conditioned_neutral", "conditional_title", "conditioned_title",
                           "pair_neutral", "pair_title")},
            "conditioned_oracle_first": {a: sum(r["oracle"][a] for r in rows) for a in VARIANTS},
            "conditioned_vs_static": {a: paired(rows, f"conditioned_{a}", f"conditional_{a}") for a in VARIANTS},
            "conditioned_vs_joint": {a: paired(rows, f"conditioned_{a}", f"pair_{a}") for a in VARIANTS}}


def evaluate(static_path, conditioned_path, model_path, reference_path):
    rows, dynamic, reference, metadata = load(static_path, conditioned_path, reference_path)
    heads = json.loads(model_path.read_text())
    pretrained = heads["conditioned_provenance"]
    assert all(pretrained[k] == metadata[k] for k in ("assets", "baseline_sha256", "config_sha256"))
    assert pretrained["source_sha256"]["articles"] == metadata["source_sha256"]["articles"]
    matrix, (starts, _, _, counts) = design(rows)
    results = []
    for row, item, old, start, count in zip(rows, dynamic, reference, starts, counts, strict=True):
        i, j = orders(row)
        gold = set(row["gold"])
        first = old["first"]
        selected = {key: old["selected"][key] for key in ("conditional_neutral", "conditional_title",
                                                         "pair_neutral", "pair_title")}
        oracle = {}
        for variant in VARIANTS:
            full = matrix[f"pair_{variant}"][start:start + count]
            estimates = {}
            for anchor in sorted({first} | gold):
                ids, extra = conditioned_branch(i, j, anchor, item["conditioned"])
                score = linear(np.column_stack((full[ids], extra)), heads["models"][variant])
                match = int(ids[score.argmax()])
                partner = int(j[match] if i[match] == anchor else i[match])
                estimates[anchor] = [anchor, partner]
            selected[f"conditioned_{variant}"] = estimates[first]
            oracle[variant] = any(set(estimates[a]) == gold for a in gold)
        success = {k: set(p) == gold for k, p in selected.items()}
        assert all(success[k] == old["success"][k] for k in success if k in old["success"])
        results.append({"id": row["id"], "source_id": row["source_id"], "first": first,
                        "first_gold": first in gold, "gold_title_link": old["gold_title_link"],
                        "gold": row["gold"], "selected": selected, "oracle": oracle, "success": success})
    subsets = {"all": results, "first_gold": [r for r in results if r["first_gold"]],
               "title_link": [r for r in results if r["gold_title_link"]],
               "no_title_link": [r for r in results if not r["gold_title_link"]]}
    return {"conditioned_provenance": metadata, "reference_sha256": digest(reference_path),
            "dev_static_sha256": digest(static_path), "dev_conditioned_sha256": digest(conditioned_path),
            "fit_sha256": digest(model_path), "cohorts": {k: summarize(v) for k, v in subsets.items()},
            "rows": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="mode", required=True)
    for cmd, inputs in (("fit", ("static", "conditioned", "output")),
                        ("evaluate", ("static", "conditioned", "model", "reference", "output"))):
        child = commands.add_parser(cmd)
        for name in inputs:
            child.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = fit(a.static, a.conditioned) if a.mode == "fit" else evaluate(a.static, a.conditioned, a.model, a.reference)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
