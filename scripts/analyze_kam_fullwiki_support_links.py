#!/usr/bin/env python3
"""Audit the paired HotpotQA link experiment and summarize its held-out strata."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import mean

ARMS = ("learned_single_shot", "learned_sequential", "one_pass_pair",
        "matched_plain", "context_only", "support_link")
PAIR_HEADS = ARMS[2:]
METRICS = ("both_support", "answer_em", "joint_em", "wrong_linked")


def value(row, arm, metric):
    r = row["arms"][arm]
    return {
        "both_support": r["both_support"],
        "answer_em": r["em"],
        "joint_em": r["both_support"] and r["em"],
        "wrong_linked": r["selected_linked"] and not r["both_support"],
    }[metric]


def paired(differences):
    n = len(differences)
    rng = Random(225479)
    estimates = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(5000))
    return {"delta": mean(differences), "bootstrap95": [estimates[125], estimates[4874]],
            "improved": sum(d > 0 for d in differences), "worsened": sum(d < 0 for d in differences)}


def summarize(fixture, fit, shards, fixture_sha):
    assert len(shards) == 4
    assert all(s["fixture_sha256"] == fixture_sha for s in [fit, *shards])
    for key in ("weights_sha256", "index_sha256", "config_sha256", "models"):
        assert all(s[key] == fit[key] for s in shards)
    assert len({shard["checkpoint_sha256"] for shard in shards}) == 1
    train, test = (fixture["rows"][group] for group in ("training", "test"))
    train_ids = {r["id"] for r in train}
    assert len(train_ids) == len(train) and not train_ids.intersection(q["id"] for q in test)
    test_gold = {i for q in test for i in q["gold_ids"] if i is not None}
    assert [r["id"] for r in fit["rows"]] == [r["id"] for r in train if None not in r["gold_ids"]]
    assert all(not test_gold.intersection(row["row_ids"]) for row in fit["rows"])
    assert sum(r["gold_linked"] and r["linked_pairs"] > 1 for r in fit["rows"]) == fit["linked_training_with_distractor"]
    rows = []
    for i, shard in enumerate(shards):
        expected = test[i::len(shards)]
        assert [r["id"] for r in shard["rows"]] == [q["id"] for q in expected]
        for row, case in zip(shard["rows"], expected, strict=True):
            pool = row["row_ids"]
            assert len(set(pool)) == len(pool) == fixture["test_depth"]
            gold = {slot for slot, idx in enumerate(pool) if idx in case["gold_ids"]}
            assert gold == set(row["gold_slots"])
            assert set(row["arms"]) == set(ARMS)
            for record in row["arms"].values():
                assert len(set(record["pair"])) == 2
                assert all(0 <= slot < len(pool) for slot in record["pair"])
                assert record["both_support"] == (gold == set(record["pair"]))
                if record["both_support"]:
                    assert record["selected_linked"] == row["gold_linked"]
            contest = row["gold_linked"] and row["linked_pairs"] > 1
            assert set(row["link_ranking"]) == (set(PAIR_HEADS) if contest else set())
            rows.append(row)

    groups = {
        "all": rows,
        "linked_gold": [r for r in rows if r["gold_linked"] is True],
        "unlinked_gold": [r for r in rows if r["gold_linked"] is False],
        "missing_gold": [r for r in rows if r["gold_linked"] is None],
        "linked_gold_with_distractors": [r for r in rows if r["link_ranking"]],
    }
    results = {}
    for group, subset in groups.items():
        results[group] = {
            "n": len(subset),
            "counts": {arm: {metric: sum(value(r, arm, metric) for r in subset) for metric in METRICS}
                       for arm in ARMS},
            "support_link_differences": {
                arm: {metric: paired([value(r, "support_link", metric) - value(r, arm, metric) for r in subset])
                      for metric in ("both_support", "joint_em")}
                for arm in ARMS[:-1]
            } if subset else {},
        }
    challenge = groups["linked_gold_with_distractors"]
    direct = {}
    for arm in PAIR_HEADS:
        outcomes = [r["link_ranking"][arm] for r in challenge]
        direct[arm] = {
            "top": sum(r["top"] for r in outcomes),
            "mean_pairwise_win": mean(r["pairwise_win"] for r in outcomes),
            "difference_vs_support_link": paired([r["link_ranking"]["support_link"]["top"] - r["link_ranking"][arm]["top"]
                                                  for r in challenge]),
        }
    return {"groups": results, "linked_only_ranking": direct,
            "linked_distractor_pairs": sum(r["linked_pairs"] - 1 for r in challenge),
            "training": {"n": len(fit["rows"]), "linked_with_distractors": fit["linked_training_with_distractor"],
                         "both_support_before_insertion": sum(r["naturally_retrieved_gold"] == 2 for r in fit["rows"])},
            "gpu": {"fit": fit["device"], "eval": [s["device"] for s in shards],
                    "fit_elapsed_seconds": fit["elapsed_seconds"],
                    "eval_elapsed_seconds": [s["elapsed_seconds"] for s in shards],
                    "eval_peak_allocated_bytes": [s["peak_allocated_bytes"] for s in shards]},
            "models": fit["models"], "fixture_sha256": fixture_sha,
            "weights_sha256": fit["weights_sha256"], "checkpoint_sha256": shards[0]["checkpoint_sha256"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    fixture = (a.directory / "fixture.json").read_bytes()
    fit = json.loads((a.directory / "fit-0.json").read_text())
    shards = [json.loads((a.directory / f"eval-{i}.json").read_text()) for i in range(4)]
    result = summarize(json.loads(fixture), fit, shards, sha256(fixture).hexdigest())
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if a.output:
        a.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
