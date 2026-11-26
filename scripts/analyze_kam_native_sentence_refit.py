#!/usr/bin/env python3
"""Fit Wikipedia-pool rerankers before scoring a disjoint native HotpotQA cohort."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analyze_kam_sentence_verifier import SENTENCE_FEATURE_GROUPS, fit, paired

ARMS = ("frozen_context", "frozen_no_coverage", "transferred_2wiki", "transferred_2wiki_clean", "native_base", "native_relevance",
        "native_mentions", "native_semantic", "native_linked_semantic", "native_semantic_shortlist4")


def fit_native(training, columns):
    groups = {
        "native_base": list(range(5)),
        "native_relevance": list(range(5)) + [columns.index(c) for c in SENTENCE_FEATURE_GROUPS["relevance_without_mentions"]],
        "native_mentions": list(range(5)) + [columns.index(c) for c in SENTENCE_FEATURE_GROUPS["mention_only"]],
        "native_semantic": list(range(len(columns))),
        "native_linked_semantic": list(range(len(columns))),
    }
    models = {}
    for name, indices in groups.items():
        examples = training
        if name == "native_linked_semantic":
            examples = [r for r in examples if r["gold_linked"] is True and r["linked_pairs"] > 1]
        projected = []
        for r in examples:
            case = r["variants"]["clean"]
            inputs = [[row[i] for i in indices] for row in case["features"]]
            projected.append({"variants": {"clean": case | {"features": inputs}}})
        center, scale, coefficients, count = fit(projected, len(indices), ("clean",))
        models[name] = {"columns": indices, "fit_cases": count, "mean": center.tolist(),
                        "scale": scale.tolist(), "coefficients": coefficients.tolist()}
    return models


def evaluate(row, models):
    case = row["variants"]["clean"]
    x = np.asarray(case["features"], dtype=np.float64)
    gold = case["gold"]
    choices = dict(zip(ARMS[:2], case["winners"], strict=True))
    logits = {}
    for name, model in models.items():
        logits[name] = ((x[:, model["columns"]] - model["mean"]) / model["scale"]) @ model["coefficients"]
        choices[name] = int(logits[name].argmax())
    subset = np.unique(np.concatenate((np.argsort(-x[:, :2], axis=0, kind="stable")[:4].ravel(), case["winners"])))
    choices["native_semantic_shortlist4"] = int(subset[logits["native_semantic"][subset].argmax()])
    linked = np.flatnonzero(x[:, 4])
    linked_scores = {"frozen_context": x[:, 0], "frozen_no_coverage": x[:, 1], **logits}
    return {
        "correct": {name: choices[name] == gold for name in ARMS},
        "wrong_link": {name: int(x[choices[name], 4]) == 1 and choices[name] != gold for name in ARMS},
        "supports": {name: row["shortlist_support_counts"][choices[name]] for name in ARMS},
        "ranked_linked_only": {name: bool(len(linked) and linked[values[linked].argmax()] == gold)
                               for name, values in linked_scores.items()},
        "gold_in_shortlist": gold >= 0,
    }


def reader_summary(shards, heldout, panels, decisions):
    rows = {r["id"]: r for s in shards for r in s["rows"]}
    assert len(rows) == sum(len(s["rows"]) for s in shards) == len(heldout)
    assert all(s["fitted_sha256"] == shards[0]["fitted_sha256"] for s in shards)
    assert all(s["old_policy_sha256"] == shards[0]["old_policy_sha256"] for s in shards)
    names = tuple(next(iter(rows.values()))["arms"])
    matched = set()
    for prior in heldout:
        id_ = prior["id"]
        current = rows[id_]
        if prior["pool_sha256"] == current["pool_sha256"]:
            assert all(current["arms"][k]["both_support"] == decisions[id_]["correct"][k] for k in names), id_
            matched.add(id_)
    output = {}
    for label, full_cohort in panels.items():
        cohort = [r for r in full_cohort if r["id"] in matched]
        per_model = {}
        for name in names:
            results = [rows[r["id"]]["arms"][name] for r in cohort]
            per_model[name] = {"answer_exact": [r["answer_exact"] for r in results],
                               "joint_exact": [r["both_support"] and r["answer_exact"] for r in results]}
        output[label] = {
            "n": len(cohort),
            "arms": {name: {k: sum(v) for k, v in metrics.items()} for name, metrics in per_model.items()},
            "versus_frozen_context": {name: {k: paired(v, per_model["frozen_context"][k]) for k, v in metrics.items()}
                                     for name, metrics in per_model.items() if name != "frozen_context"},
        }
    return {"fitted_sha256": shards[0]["fitted_sha256"], "old_policy_sha256": shards[0]["old_policy_sha256"],
            "unmatched_ranked_pool_ids": sorted(rows.keys() - matched),
            "gpu": sorted({s["gpu"] for s in shards}), "tests": output}


def exploratory_weight_swaps(panels, models, columns):
    def anchored(model):
        w = np.asarray(model["coefficients"]) / np.asarray(model["scale"])
        assert w[0] > 0
        return w / w[0]

    transferred = anchored(models["transferred_2wiki"])
    native = anchored(models["native_semantic"])
    groups = {
        "pair_features": list(range(5)),
        "sentence_relevance": [columns.index(c) for c in SENTENCE_FEATURE_GROUPS["relevance_without_mentions"]],
        "mention_local": [columns.index(c) for c in SENTENCE_FEATURE_GROUPS["mention_only"]],
    }
    result = {}
    for key in ("natural_linked_distractor", "gold_unlinked"):
        cases = [row["variants"]["clean"] for row in panels[key]]

        def hits(w):
            return [int((np.asarray(case["features"]) @ w).argmax()) == case["gold"] for case in cases]

        before = hits(transferred)
        counts = {"transferred": sum(before), "native": sum(hits(native))}
        for group, indices in groups.items():
            weights = transferred.copy()
            weights[indices] = native[indices]
            counts[group] = sum(hits(weights))
        result[key] = {"n": len(cases), "correct": counts}
    return result


def summarize(shards, old_policy, old_fit, audit, reader):
    fields = ("fixture_sha256", "index_sha256", "heads_sha256", "baseline_sha256", "ablations_sha256", "columns", "shards")
    assert all(all(s[k] == shards[0][k] for k in fields) for s in shards)
    assert audit["fixture_sha256"] == shards[0]["fixture_sha256"]
    assert {s["shard"] for s in shards} == set(range(shards[0]["shards"]))
    rows = sorted([r for s in shards for r in s["rows"]], key=lambda r: r["id"])
    assert len({r["id"] for r in rows}) == len(rows)
    assert Counter(r["split"] for r in rows) == {"training": 1536, "test": 1024}
    training = [r for r in rows if r["split"] == "training"]
    heldout = [r for r in rows if r["split"] == "test"]
    overlap_ids = set(audit["test_ids_sharing_transferred_fit_titles"])
    assert overlap_ids <= {r["id"] for r in heldout}
    columns = shards[0]["columns"]
    assert old_policy["features"] == columns
    assert all(s["columns"] == columns and s["fixture_sha256"] == old_policy["fixture_sha256"] for s in old_fit)
    transferred_training = sorted(
        (r for s in old_fit for r in s["rows"] if r["split"] == "twowiki-fit"), key=lambda r: r["id"],
    )
    assert len(transferred_training) == 768
    old_center, old_scale, old_weights, old_count = fit(transferred_training, len(columns), ("clean",))
    transferred_clean = {"columns": list(range(len(columns))), "mean": old_center.tolist(),
                         "scale": old_scale.tolist(), "coefficients": old_weights.tolist()}
    fitted = fit_native(training, columns)
    models = {**fitted, "transferred_2wiki": {"columns": list(range(len(columns))), **old_policy},
              "transferred_2wiki_clean": transferred_clean}
    decisions = {r["id"]: evaluate(r, models) for r in heldout}

    panels = {
        "all": heldout,
        "gold_in_pool": [r for r in heldout if r["gold_in_pool"] == 2],
        "gold_linked": [r for r in heldout if r["gold_linked"] is True],
        "natural_linked_distractor": [r for r in heldout if r["gold_linked"] is True and r["linked_pairs"] > 1],
        "gold_unlinked": [r for r in heldout if r["gold_linked"] is False],
        "missing_gold": [r for r in heldout if r["gold_in_pool"] < 2],
    }
    primary = panels["natural_linked_distractor"]
    panels["natural_linked_distractor_without_old_fit_titles"] = [r for r in primary if r["id"] not in overlap_ids]
    panels["linked_distractor_shares_support_in_shortlist"] = [
        r for r in primary if any(f[4] and count == 1 for f, count in zip(
            r["variants"]["clean"]["features"], r["shortlist_support_counts"], strict=True))
    ]
    results = {}
    for group, cohort in panels.items():
        panel = [decisions[r["id"]] for r in cohort]
        responses = {arm: [r["correct"][arm] for r in panel] for arm in ARMS}
        results[group] = {
            "n": len(panel), "gold_in_shortlist": sum(r["gold_in_shortlist"] for r in panel),
            "arms": {
                arm: {"both_support": sum(responses[arm]), "wrong_link": sum(r["wrong_link"][arm] for r in panel),
                      "wrong_link_shares_support": sum(r["wrong_link"][arm] and r["supports"][arm] == 1 for r in panel),
                      "support_count": dict(sorted(Counter(r["supports"][arm] for r in panel).items())),
                      "linked_only_gold_first": sum(r["ranked_linked_only"].get(arm, False) for r in panel)
                      if arm != "native_semantic_shortlist4" else None}
                for arm in ARMS
            },
            "vs_frozen_context": {a: paired(responses[a], responses["frozen_context"]) for a in ARMS[1:]},
            "semantic_vs_base": paired(responses["native_semantic"], responses["native_base"]),
            "semantic_vs_transferred": paired(responses["native_semantic"], responses["transferred_2wiki"]),
            "semantic_vs_clean_transferred": paired(responses["native_semantic"], responses["transferred_2wiki_clean"]),
            "clean_transfer_vs_paired_transfer": paired(responses["transferred_2wiki_clean"], responses["transferred_2wiki"]),
        }
    return {**{k: shards[0][k] for k in fields}, "gpu": sorted({s["gpu"] for s in shards}),
            "training": {"n": len(training), "gold_in_pool": sum(r["gold_in_pool"] == 2 for r in training),
                         "native_linked_distractor": sum(r["gold_linked"] is True and r["linked_pairs"] > 1 for r in training),
                         "fit_cases": {k: m["fit_cases"] for k, m in fitted.items()},
                         "transferred_clean_fit_cases": old_count},
            "tests": results, "exploratory_native_weight_swaps": exploratory_weight_swaps(panels, models, columns),
            **({"reader": reader_summary(reader, heldout, panels, decisions)} if reader else {})}, fitted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--old-policy", type=Path, required=True)
    p.add_argument("--old-fit-replay", type=Path, required=True)
    p.add_argument("--cohort-audit", type=Path, required=True)
    p.add_argument("--reader", type=Path, help="Directory of frozen-reader GPU outputs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--write-replay", type=Path)
    p.add_argument("--write-fit", type=Path)
    a = p.parse_args()
    reader = []
    if a.input.is_dir():
        shards = [json.loads(gzip.decompress(p.read_bytes())) for p in sorted(a.input.glob("features-*.json.gz"))]
    else:
        replay = json.loads(gzip.decompress(a.input.read_bytes()))
        if isinstance(replay, list):
            shards = replay
        else:
            shards, reader = replay["features"], replay["reader"]
    if a.reader:
        reader = [json.loads(gzip.decompress(p.read_bytes())) for p in sorted(a.reader.glob("reader-*.json.gz"))]
        assert reader
    assert all(s["fixture_sha256"] == shards[0]["fixture_sha256"] and s["index_sha256"] == shards[0]["index_sha256"]
               for s in reader)
    old_fit = a.old_fit_replay.read_bytes()
    summary, models = summarize(shards, json.loads(a.old_policy.read_bytes()),
                                json.loads(gzip.decompress(old_fit)), json.loads(a.cohort_audit.read_bytes()), reader)
    summary["old_policy_sha256"] = sha256(a.old_policy.read_bytes()).hexdigest()
    summary["old_fit_replay_sha256"] = sha256(old_fit).hexdigest()
    a.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if a.write_fit:
        a.write_fit.write_text(json.dumps(models, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        data = {"features": shards, "reader": reader}
        a.write_replay.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
