#!/usr/bin/env python3
"""Summarize fixed-reader answers and single-request verifier timings."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_kam_sentence_verifier import SCENARIOS, paired

ARMS = ("full", "sentence16", "sentence4")


def profiles(shards):
    results = {}
    for shard in sorted(shards, key=lambda s: (s["device"], s["dataset"])):
        rows = shard["rows"]
        key = f'{shard["device"]}:{shard["dataset"]}'
        modes = ("cold", "resident")
        if shard.get("host_cache"):
            key += ":host-cache-profile"
            modes += ("host_staged",)
        assert key not in results
        assert len({r["id"] for r in rows}) == 48
        result = {}
        for depth in (4, 8, 16):
            panel = [r for r in rows if r["depth"] == depth]
            assert len(panel) == 48
            timing = {}
            for mode in modes:
                a = [r[mode + "_ms"] for r in panel]
                timing[mode] = {"p50_ms": round(float(np.median(a)), 3),
                                "p95_ms": round(float(np.quantile(a, .95)), 3)}
            result[str(depth)] = {"requests": len(panel), "timings": timing,
                                  "mean_pairs": round(float(np.mean([r["pairs"] for r in panel])), 3),
                                  "mean_unique_sentences": round(float(np.mean([r["unique_sentences"] for r in panel])), 3),
                                  "max_abs_feature_delta": max(r["feature_max_abs_delta"] for r in panel)}
        results[key] = {"torch": shard["torch"], "resident_cache_sentences": shard["resident_cache_sentences"],
                        "vector_bytes": shard["vector_bytes"], "depths": result}
    return results


def readers(shards):
    assert len({s["policy_sha256"] for s in shards}) == len({s["fixture_sha256"] for s in shards}) == 1
    groups = defaultdict(list)
    ids = set()
    hardware = defaultdict(set)
    for shard in shards:
        for row in shard["rows"]:
            split = row["split"]
            id_ = row["id"]
            assert id_ not in ids
            ids.add(id_)
            hardware[split].add(shard["device"])
            groups[split].append(row)
    assert {key: len(values) for key, values in groups.items()} == {"twowiki-test": 420, "musique-test": 300}
    results = {}
    for split, rows in sorted(groups.items()):
        for mode in SCENARIOS:
            cases = [row["variants"][mode] for row in sorted(rows, key=lambda r: r["id"]) if mode in row["variants"]]
            outcomes = {arm: {metric: [] for metric in ("evidence", "exact", "joint")} for arm in ARMS}
            for case in cases:
                assert set(case) == set(ARMS)
                for arm in ARMS:
                    hit = case[arm]["evidence"]
                    exact = case[arm]["exact"]
                    for name, value in (("evidence", hit), ("exact", exact), ("joint", hit and exact)):
                        outcomes[arm][name].append(int(value))
            results[f"{split}:{mode}"] = {
                "n": len(cases), "hardware": sorted(hardware[split]),
                "correct": {arm: {metric: sum(v) for metric, v in m.items()} for arm, m in outcomes.items()},
                "paired_vs_full": {arm: {metric: paired(values, outcomes["full"][metric])
                                          for metric, values in outcomes[arm].items()}
                                   for arm in ARMS[1:]},
            }
    return {"policy_sha256": shards[0]["policy_sha256"], "fixture_sha256": shards[0]["fixture_sha256"],
            "tests": results}


def reader_agreement(shards, replication):
    def records(items):
        return {(r["split"], r["id"], mode, arm): decision
                for item in items for r in item["rows"] for mode, arms in r["variants"].items()
                for arm, decision in arms.items()}

    assert len({p["policy_sha256"] for p in shards + replication}) == 1
    ref, replica = records(shards), records(replication)
    assert ref.keys() == replica.keys()
    output = {}
    for split in ("twowiki-test", "musique-test"):
        keys = [k for k in ref if k[0] == split]
        output[split] = {"decisions": len(keys)}
        for field in ("evidence", "exact"):
            output[split][field + "_changed"] = sum(ref[k][field] != replica[k][field] for k in keys)
    return output


def fullwiki(shards):
    assert len({s["policy_sha256"] for s in shards}) == len({s["fixture_sha256"] for s in shards}) == 1
    rows = sorted([r for s in shards for r in s["rows"]], key=lambda r: r["id"])
    assert len(rows) == len({r["id"] for r in rows}) == 768
    panels = {
        "all": rows,
        "gold_in_pool": [r for r in rows if r["gold_linked"] is not None],
        "gold_linked": [r for r in rows if r["gold_linked"] is True],
        "gold_unlinked": [r for r in rows if r["gold_linked"] is False],
        "missing_gold": [r for r in rows if r["gold_linked"] is None],
        "gold_linked_with_distractor": [r for r in rows if r["gold_linked"] is True and r["linked_pairs"] > 1],
    }
    assert [len(panels[key]) for key in panels] == [768, 468, 368, 100, 300, 360]
    results = {}
    for label, cohort in panels.items():
        output = {}
        for arm in ARMS:
            results_by_row = [r["variants"]["clean"][arm] for r in cohort]
            output[arm] = {"evidence": [r["evidence"] for r in results_by_row],
                           "exact": [r["exact"] for r in results_by_row],
                           "joint": [r["evidence"] and r["exact"] for r in results_by_row]}
        results[label] = {"n": len(cohort), "correct": {a: {k: sum(v) for k, v in m.items()} for a, m in output.items()},
                          "paired_vs_full": {a: {k: paired(v, output["full"][k]) for k, v in output[a].items()}
                                             for a in ARMS[1:]}}
    return {"fixture_sha256": shards[0]["fixture_sha256"], "policy_sha256": shards[0]["policy_sha256"],
            "hardware": sorted({s["device"] for s in shards}),
            "previous_baseline_evidence_matches": sum(r["matches_previous_evidence"] for r in rows),
            "previous_baseline_exact_matches": sum(r["matches_previous_exact"] for r in rows),
            "tests": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Compact replay with profile and/or reader shards")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.input.read_text())
    summary = {}
    if data.get("profile"):
        summary["profile"] = profiles(data["profile"])
    if data.get("reader"):
        summary["reader"] = readers(data["reader"])
    if data.get("reader_local"):
        summary["reader_hardware_check"] = reader_agreement(data["reader"], data["reader_local"])
    if data.get("fullwiki"):
        summary["fullwiki"] = fullwiki(data["fullwiki"])
    a.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
