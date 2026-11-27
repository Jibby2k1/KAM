#!/usr/bin/env python3
"""Compare matched request groups at multiple FullWiki serving widths."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analyze_kam_shared_serving import unpack

WIDTHS = (1, 4, 8, 16, 32)
ARMS = {f"optimized/{i}": i for i in WIDTHS} | {"reference/8": 8}
STAGES = ("request_ms", "initial_search_ms", "reader_ms")
REFERENCE = "optimized/8"


def measure(rows, fixture):
    policies = {k: {phase: [] for phase in STAGES} for k in ARMS}
    group_latency = {k: [] for k in ARMS}
    allocation = {k: [] for k in ARMS}
    parity = {k: Counter() for k in ARMS}
    flags = {k: Counter() for k in ARMS}
    distinct = set()
    for index, trace in enumerate(rows):
        assert trace["shard"] == index and trace["shards"] == 4 and trace["gpu"] == rows[0]["gpu"]
        assert trace["batch_policies"] == rows[0]["batch_policies"]
        assert all(trace[k] == rows[0][k] for k in ("index_sha256", "fitted_sha256", "corpus_profile_partition", "models"))
        assert len(trace["rows"]) == 6
        for cell in trace["rows"]:
            assert len(cell["ids"]) == 32 and distinct.isdisjoint(cell["ids"])
            distinct.update(cell["ids"])
            assert set(cell["order"]) == set(cell["batches"]) == ARMS.keys()
            ref = cell["batches"][REFERENCE]["rows"]
            for policy, width in ARMS.items():
                arm = cell["batches"][policy]
                assert len(arm["timing"]) == 32 // width
                assert all(x["size"] == width for x in arm["timing"])
                for phase in STAGES:
                    policies[policy][phase] += [x[phase] for x in arm["timing"]]
                group_latency[policy].append(sum(x["request_ms"] for x in arm["timing"]))
                allocation[policy].append(arm["extra_peak_bytes"])
                for actual, candidate, old in zip(cell["ids"], arm["rows"], ref, strict=True):
                    assert actual == candidate["id"] == old["id"]
                    gold = fixture[actual]
                    ranking, control = [r["arms"]["conditioned_union"] for r in (candidate, old)]
                    support = set(ranking["pair"]) == gold
                    exact = ranking["exact"]
                    flags[policy].update(support=support, answer=exact, joint=support and exact)
                    parity[policy].update(
                        initial_set=set(candidate["initial_ids"]) == set(old["initial_ids"]),
                        initial_order=candidate["initial_ids"] == old["initial_ids"],
                        second_hop=candidate["corpus_hits"] == old["corpus_hits"],
                        selected_pair=ranking["pair"] == control["pair"], answer_flag=exact == control["exact"],
                        support_flag=support == (set(control["pair"]) == gold),
                        joint_flag=(support and exact) == (set(control["pair"]) == gold and control["exact"]),
                        complete_record=candidate == old,
                    )
    assert len(distinct) == 768
    return policies, group_latency, allocation, parity, flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--measurement", type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raw = gzip.decompress(a.fixture.read_bytes())
    seal = sha256(raw).hexdigest()
    traces = sorted(map(unpack, a.measurement), key=lambda t: t["shard"])
    assert all(t["fixture_sha256"] == seal and set(t["batch_policies"]) == ARMS.keys() for t in traces)
    fixture = {q["id"]: set(q["gold_ids"]) for q in json.loads(raw)["rows"]["test"]}
    stages, groups, allocation, parity, flags = measure(traces, fixture)
    count = len(groups[REFERENCE])
    sample = np.random.default_rng(829539).integers(0, count, size=(5000, count))
    results = {}
    for arm in ARMS:
        time = np.array(groups[arm])
        other = time - np.asarray(groups[REFERENCE])
        estimate = np.median(other[sample], axis=1)
        results[arm] = {
            "requests_per_second": 32 * count * 1000 / float(sum(time)),
            "p50_p95_group_ms": np.quantile(time, [.5, .95]).tolist(),
            "median_group_delta_ms": float(np.median(other)),
            "median_group_delta_95_ms": np.quantile(estimate, [.025, .975]).tolist(),
            "p50_p95_subbatch_ms": {k: np.quantile(seq, [.5, .95]).tolist() for k, seq in stages[arm].items()},
            "max_extra_peak_bytes": max(allocation[arm]),
            "quality_counts": flags[arm],
            "parity_vs_optimized_8": parity[arm],
        }
    result = {"fixture_sha256": seal, "device": traces[0]["gpu"], "requests": 768, "groups": count, "policies": results}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
