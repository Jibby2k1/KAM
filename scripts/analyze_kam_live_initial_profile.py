#!/usr/bin/env python3
"""Audit matched live dense and hybrid HotpotQA request timings."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import median

from analyze_kam_shared_serving import metrics, unpack

ARMS = ("dense", "hybrid")
WARM_ARMS = ("dense", "hybrid_serial", "hybrid_two_threads")
PHASES = ("initial_search_ms", "anchor_ms", "conditioned_search_ms", "remaining_rerank_ms", "reader_ms")
FROZEN = ("fixture_sha256", "config_sha256", "index_sha256", "fitted_sha256", "heads_sha256",
          "ablations_sha256", "baseline_sha256", "models", "guard_columns", "split")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def examine(shards, study, warm=False):
    arms = WARM_ARMS if warm else ARMS
    routes = {arm: "hybrid" if arm.startswith("hybrid") else arm for arm in arms}
    fixture = gzip.decompress((study / "fixture.json.gz").read_bytes())
    questions = json.loads(fixture)["rows"]["test"]
    ids = [r["id"] for r in questions]
    gold = {r["id"]: set(r["gold_ids"]) for r in questions}
    assert len(ids) == len(gold) == 1024
    assert [s["shard"] for s in shards] == list(range(4))
    assert all(s["gpu"] == "NVIDIA B200" and s["shards"] == 4 and s["prewarm_seconds"] > 0 for s in shards)
    assert len({s["database_bytes"] for s in shards}) == 1
    frozen = {k: shards[0][k] for k in FROZEN}
    assert frozen["fixture_sha256"] == sha256(fixture).hexdigest()
    assert all(all(s[k] == v for k, v in frozen.items()) for s in shards)
    pools, archives, refs = {}, {}, {}
    for method in ARMS:
        pool_path = study / f"pools-{method}.json.gz"
        replay = study / f"{method}-broad-replay.json.gz"
        sealed, records = unpack(pool_path), unpack(replay)
        refs[method] = {"pool_sha256": digest(pool_path), "replay_sha256": digest(replay)}
        assert sealed["method"] == method and sealed["pool_depth"] == 64 and sealed["component_depth"] == 128
        assert all(sealed[k] == frozen[k] for k in ("fixture_sha256", "index_sha256", "config_sha256"))
        assert all(s["pools_sha256"] == refs[method]["pool_sha256"] for s in records)
        assert len(records) == 4 and [s["shard"] for s in records] == list(range(4))
        assert all(s["gpu"] == "NVIDIA B200" and all(s[k] == v for k, v in frozen.items()) for s in records)
        pools[method] = sealed["rows"]
        archives[method] = {r["id"]: r for s in records for r in s["rows"]}
        assert set(ids) == pools[method].keys() == archives[method].keys()
    assert all(s["pools_sha256"] == refs["dense"]["pool_sha256"] for s in shards)
    fields = ("initial_order", "initial_membership", "anchor", "expanded_order", "expanded_membership", "arms")
    parity = {m: dict.fromkeys(fields, 0) for m in arms}
    outcomes = {m: dict.fromkeys(("coverage", "support", "joint_em"), 0) for m in arms}
    parallel_parity = dict.fromkeys(("initial_ids", "first", "corpus_hits", "arms", "features"), 0)
    rows = []
    for shard in shards:
        sample = shard["rows"]
        assert [r["id"] for r in sample] == ids[shard["shard"]::4][4:132:4]
        rng = Random(23847 + shard["shard"])
        for row in sample:
            run_order = list(arms)
            rng.shuffle(run_order)
            assert row["order"] == run_order
            id_ = row["id"]
            for arm, method in routes.items():
                new, old = row["records"][arm], archives[method][id_]
                assert new["id"] == id_
                initial = new["initial_ids"]
                expected = pools[method][id_]
                assert len(initial) == len(set(initial)) == 64
                assert len(expected) == len(set(expected)) == 64
                assert new["pool_hash"] == sha256(json.dumps(initial, separators=(",", ":")).encode()).hexdigest()
                assert old["pool_hash"] == sha256(json.dumps(expected, separators=(",", ":")).encode()).hexdigest()
                pairs = new["arms"]
                assert new["first"] == pairs["broad"]["pair"][0]
                phase = row["phase_ms"][arm]
                assert set(phase) == set(PHASES) and all(t > 0 for t in phase.values())
                assert sum(phase.values()) <= row["timing_ms"][arm]
                hits, prior = (r["corpus_hits"]["conditioned_union"] for r in (new, old))
                assert len(hits) == len(set(hits)) == 64
                matches = (initial == expected, set(initial) == set(expected), new["first"] == old["first"],
                           hits == prior, set(hits) == set(prior), pairs == old["arms"])
                for k, equal in zip(fields, matches, strict=True):
                    parity[arm][k] += equal
                coverage = set(initial + hits)
                assert set(pairs["broad"]["pair"]) <= set(initial)
                selected = set(pairs["conditioned_union"]["pair"])
                assert selected <= coverage and len(selected) == 2
                outcomes[arm]["coverage"] += gold[id_] <= coverage
                recovered = selected == gold[id_]
                outcomes[arm]["support"] += recovered
                outcomes[arm]["joint_em"] += recovered and pairs["conditioned_union"]["exact"]
            if warm:
                serial, threaded = (row["records"][arm] for arm in WARM_ARMS[1:])
                for k in parallel_parity:
                    parallel_parity[k] += serial[k] == threaded[k]
        rows.extend(sample)

    def differences(subset, times):
        paired = [{"serving_timing_ms": {"reference": r[times]["dense"], "shared": r[times]["hybrid"]}} for r in subset]
        t = metrics(paired)
        return {"questions": len(paired), "dense": t["reference"], "hybrid": t["shared"],
                "hybrid_overhead_ms": -t["paired_median_saving_ms"],
                "hybrid_overhead_ci95_ms": [-x for x in reversed(t["paired_median_saving_ci95_ms"])],
                "hybrid_faster": t["shared_faster"]}

    common = {"phases": {m: {k: round(median(r["phase_ms"][m][k] for r in rows), 6) for k in PHASES}
                         for m in arms},
              "archive_parity": parity, "live_quality": outcomes, "database_bytes": shards[0]["database_bytes"],
              "prewarm_seconds": [round(s["prewarm_seconds"], 6) for s in shards]}
    if not warm:
        return {**frozen, "references": refs, "all": differences(rows, "timing_ms"),
                "by_order": {m + "_first": differences([r for r in rows if r["order"][0] == m], "timing_ms")
                             for m in ARMS}, **common}

    def contrast(first, second, time_key):
        samples = [{"serving_timing_ms": {"reference": row["timing_ms"][first] if not time_key else
                                         row["phase_ms"][first][time_key],
                                         "shared": row["timing_ms"][second] if not time_key else
                                         row["phase_ms"][second][time_key]}} for row in rows]
        result = metrics(samples)
        return {first: result.pop("reference"), second: result.pop("shared"), **result}

    assert len({s["lexical_prewarm_bytes"] for s in shards}) == 1
    assert all(s["lexical_prewarm_seconds"] > 0 for s in shards)
    compare = (("hybrid_serial", "hybrid_two_threads"), ("dense", "hybrid_two_threads"))
    return {**frozen, "references": refs, **common, "parallel_exact_parity": parallel_parity,
            "lexical_prewarm_bytes": shards[0]["lexical_prewarm_bytes"],
            "lexical_prewarm_seconds": [round(s["lexical_prewarm_seconds"], 6) for s in shards],
            "comparisons": {f"{b}_vs_{a}": {"request": contrast(a, b, None),
                                            "initial_search": contrast(a, b, "initial_search_ms")}
                            for a, b in compare}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", type=Path, nargs=4)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warm-lexical", action="store_true")
    a = p.parse_args()
    result = examine([unpack(f) for f in a.profiles], a.study, a.warm_lexical)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
