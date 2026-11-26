#!/usr/bin/env python3
"""Audit paired HotpotQA outcomes and resample shared-support document groups."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random
from statistics import mean

from audit_kam_natural_data import digest
from prepare_kam_natural_router import title_key

ARMS = ("frozen_single_shot", "frozen_sequential", "learned_single_shot", "learned_sequential", "gold")
METRICS = {
    "both_support": lambda r: r["both_support"],
    "exact_match": lambda r: r["em"],
    "joint_exact_match": lambda r: r["em"] and r["both_support"],
    "token_f1": lambda r: r["f1"],
}


def groups_for(source, ids):
    selected = set(ids)
    raw = {r["_id"]: r for r in json.loads(source.read_text()) if r["_id"] in selected}
    parents = {}

    def root(title):
        parents.setdefault(title, title)
        while title != parents[title]:
            title = parents[title]
        return title

    supports = {}
    for id_ in ids:
        titles = sorted({title_key(title) for title, _ in raw[id_]["supporting_facts"]})
        assert len(titles) == 2
        supports[id_] = titles[0]
        parents[root(titles[1])] = root(titles[0])
    groups = defaultdict(list)
    for id_, title in supports.items():
        groups[root(title)].append(id_)
    return list(groups.values()), raw


def summarize(root, source, config):
    manifest = json.loads((root / "partition.json").read_text())
    ids = manifest["ids"]
    spec = json.loads(config.read_text())
    arms = spec.get("arms", ARMS)
    contrasts = spec.get("contrasts", (
        ("learned_sequential", "learned_single_shot"),
        ("learned_single_shot", "frozen_sequential"),
    ))
    assert digest(source) == manifest["source_sha256"]
    groups, cases = groups_for(source, ids)
    values = {arm: defaultdict(list) for arm in arms}
    by_seed = []
    profiles = []
    for index, seed in enumerate(spec["seeds"]):
        shard = json.loads((root / f"shard_{index}.json").read_text())
        assert shard["status"] == "completed" and shard["seed"] == seed
        assert shard["partition_sha256"] == digest(root / "partition.json")
        assert shard["config_sha256"] == digest(config)
        rows = defaultdict(dict)
        for row in shard["records"]:
            assert row["id"] not in rows[row["arm"]]
            case = cases[row["id"]]
            gold = {title for title, _ in case["supporting_facts"]}
            assert len(set(row["selected"])) == 2
            assert row["both_support"] == (gold <= {case["context"][i][0] for i in row["selected"]})
            rows[row["arm"]][row["id"]] = row
        assert set(rows) == set(arms)
        per_arm = {}
        for arm, table in rows.items():
            assert set(table) == set(ids)
            per_arm[arm] = {metric: mean(fn(row) for row in table.values()) for metric, fn in METRICS.items()}
            for id_, row in table.items():
                values[arm][id_].append(row)
        by_seed.append({"seed": seed, "arms": per_arm})
        profiles.append(shard["timing"])

    averaged = {arm: {metric: mean(per_seed["arms"][arm][metric] for per_seed in by_seed)
                      for metric in METRICS} for arm in arms}
    intervals = {}
    for name, reference in contrasts:
        paired = {}
        for metric, fn in METRICS.items():
            group_deltas = [sum(mean(map(fn, values[name][id_])) -
                                mean(map(fn, values[reference][id_])) for id_ in group) for group in groups]
            lengths = [len(group) for group in groups]
            rng = Random(821040)
            draws = []
            for _ in range(4000):
                sample = rng.choices(range(len(groups)), k=len(groups))
                draws.append(sum(group_deltas[i] for i in sample) / sum(lengths[i] for i in sample))
            draws.sort()
            paired[metric] = {"delta": sum(group_deltas) / len(ids),
                              "interval": [draws[99], draws[3899]]}
        intervals[name + "_minus_" + reference] = paired
    return {"n": len(ids), "document_groups": len(groups), "largest_group": max(map(len, groups)),
            "mean": averaged, "seed_results": by_seed, "comparison_intervals": intervals,
            "timing_mean_seconds": {key: mean(p[key] for p in profiles) for key in profiles[0]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/nonlinear_retrieval/natural/router_v1.json"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(summarize(a.root, a.source, a.config), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
