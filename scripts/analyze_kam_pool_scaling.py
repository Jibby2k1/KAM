#!/usr/bin/env python3
"""Audit nested pools, paired quality and batch-one timing."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random
from statistics import mean, median

from analyze_kam_hotpot_head_to_head import METRICS, groups_for
from audit_kam_natural_data import digest
from prepare_kam_natural_router import title_key

def read(path):
    return json.loads(path.read_text())


def percentile(values, q):
    data = sorted(values)
    pos = (len(data) - 1) * q
    lower = int(pos)
    return data[lower] + (data[min(lower + 1, len(data) - 1)] - data[lower]) * (pos - lower)


def paired_intervals(values, groups, spec):
    result = {}
    sizes = [len(grp) for grp in groups]
    total = sum(sizes)
    for arm, ref in spec["contrasts"]:
        outcomes = {}
        for name, metric in METRICS.items():
            deltas = [sum(mean(map(metric, values[arm][id_])) - mean(map(metric, values[ref][id_]))
                          for id_ in grp) for grp in groups]
            rng = Random(281109)
            bootstrap = []
            for _ in range(4000):
                g = rng.choices(range(len(groups)), k=len(groups))
                bootstrap.append(sum(deltas[i] for i in g) / sum(sizes[i] for i in g))
            outcomes[name] = {"delta": sum(deltas) / total,
                              "interval": [percentile(bootstrap, .025), percentile(bootstrap, .975)]}
        result[f"{arm}_minus_{ref}"] = outcomes
    return result


def analyze(root, source, config, pools=None):
    if pools is None:
        pools = root / "pools.json"
    manifest = read(pools)
    assert digest(source) == manifest["source_sha256"]
    ids, sizes = manifest["ids"], manifest["sizes"]
    groups, originals = groups_for(source, ids)
    cases = {c["_id"]: c for c in read(source)}
    spec = read(config)
    timed = [arm for arm in spec["arms"] if arm != "learned_single_shot"]
    contexts, gold = {}, {}
    for row in manifest["rows"]:
        key = row["id"]
        titles = [title_key(cases[id_]["context"][i][0]) for id_, i in row["original"] + row["added"]]
        assert len(set(titles)) == sizes[-1] == len(titles)
        contexts[key] = titles
        gold[key] = {title_key(title) for title, _ in originals[key]["supporting_facts"]}
        assert gold[key] <= set(titles[:sizes[0]])
    all_rows = {size: {arm: defaultdict(list) for arm in spec["arms"]} for size in sizes}
    times = {size: {arm: {} for arm in timed} for size in sizes}
    by_seed = []
    seed_meta = []
    rng = Random(spec["latency_seed"])
    timed_ids = {ids[i] for i in rng.sample(range(1, len(ids)), spec["latency_questions"])}
    for idx, seed in enumerate(spec["seeds"]):
        shard = read(root / f"shard_{idx}.json")
        assert (shard["status"], shard["seed"]) == ("completed", seed)
        assert shard["pools_sha256"] == digest(pools) and shard["config_sha256"] == digest(config)
        by_cell = defaultdict(dict)
        for row in shard["records"]:
            id_, size, arm = row["id"], row["pool_size"], row["arm"]
            left, right = row["selected"]
            assert 0 <= left < size and 0 <= right < size and left != right
            assert row["both_support"] == (gold[id_] <= {contexts[id_][left], contexts[id_][right]})
            assert id_ not in by_cell[(size, arm)]
            by_cell[(size, arm)][id_] = row
            all_rows[size][arm][id_].append(row)
        expected = {(n, arm) for n in sizes for arm in spec["arms"]}
        assert set(by_cell) == expected and all(set(rows) == set(ids) for rows in by_cell.values())
        seed_results = {}
        seed_meta.append((shard["device"], shard["torch"], shard["models"]))
        for (size, arm), rows in by_cell.items():
            seed_results.setdefault(str(size), {})[arm] = {
                key: mean(fn(row) for row in rows.values()) for key, fn in METRICS.items()}
        by_seed.append({"seed": seed, "sizes": seed_results, "setup_seconds": shard["setup_seconds"],
                        "elapsed_seconds": shard["elapsed_seconds"], "peak_allocated_bytes": shard["peak_allocated_bytes"]})
        for row in shard["latency"]:
            id_, size, arm = row["id"], row["pool_size"], row["arm"]
            assert id_ in timed_ids and row["total_seconds"] >= row["retrieval_seconds"] > 0
            assert (seed, id_) not in times[size][arm]
            times[size][arm][(seed, id_)] = row
    assert all(value == seed_meta[0] for value in seed_meta)
    for table in times.values():
        assert all(set(rows) == {(seed, id_) for seed in spec["seeds"] for id_ in timed_ids}
                   for rows in table.values())
    output = {}
    for size, table in all_rows.items():
        quality = {arm: {
            key: mean(mean(map(fn, rows)) for rows in grouped.values()) for key, fn in METRICS.items()
        } for arm, grouped in table.items()}
        latency = {}
        baseline = times[size]["learned_sequential"]
        for arm, rows in times[size].items():
            latency[arm] = {"n": len(rows)}
            for metric in ("total_seconds", "retrieval_seconds"):
                sample = [row[metric] for row in rows.values()]
                latency[arm][metric] = {
                    "p50": median(sample), "p95": percentile(sample, .95),
                    "paired_median_saving_vs_sequential": median(baseline[k][metric] - r[metric] for k, r in rows.items()),
                }
            latency[arm]["same_unordered_selection_as_batch"] = sum(
                set(row["selected"]) == set(next(
                    shard_row["selected"] for idx, shard_row in enumerate(table[arm][id_])
                    if spec["seeds"][idx] == seed
                )) for (seed, id_), row in rows.items()
            )
        output[str(size)] = {"quality": quality, "paired_intervals": paired_intervals(table, groups, spec),
                             "latency": latency}
    return {"n": len(ids), "shared_support_groups": len(groups), "largest_group": max(map(len, groups)),
            "by_size": output, "by_seed": by_seed, "backend": seed_meta[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/nonlinear_retrieval/natural/pool_scaling_v1.json"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(analyze(a.root, a.source, a.config), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
