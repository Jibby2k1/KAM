#!/usr/bin/env python3
"""Audit paired live anchoring and request-local sentence reuse on B200."""

import argparse
from hashlib import sha256
import gzip
import json
from pathlib import Path
from random import Random
from statistics import median

from analyze_kam_shared_serving import metrics, unpack

POLICIES = ("affinity", "live_broad")
PHASES = ("anchor_ms", "conditioned_search_ms", "remaining_rerank_ms", "reader_ms")
FIELDS = ("pool_hash", "first", "corpus_hits", "arms")


def evaluate(profiles, fixture_path, refs):
    policies = tuple(refs)
    reuse = policies == ("live_broad", "live_reuse")
    assert policies == POLICIES or reuse
    baseline, treatment = policies
    fixture = gzip.decompress(fixture_path.read_bytes())
    ids = [q["id"] for q in json.loads(fixture)["rows"]["test"]]
    assert len(ids) == len(set(ids)) == 1024
    archives = {p: unpack(refs[p]) for p in policies}
    metadata = ("fixture_sha256", "pools_sha256", "config_sha256", "index_sha256", "fitted_sha256",
                "heads_sha256", "ablations_sha256", "baseline_sha256", "models", "guard_columns", "split")
    frozen = {k: archives[baseline][0][k] for k in metadata}
    assert frozen["fixture_sha256"] == sha256(fixture).hexdigest()
    assert [s["shard"] for s in profiles] == list(range(4))
    assert all(s["gpu"] == "NVIDIA B200" and s["prewarm_seconds"] > 0 for s in profiles)
    assert len({s["database_bytes"] for s in profiles}) == 1

    expected = {}
    for policy, shards in archives.items():
        assert len(shards) == 4
        expected[policy] = {r["id"]: r for s in shards for r in s["rows"]}
        assert expected[policy].keys() == set(ids)
    assert all(all(s[k] == v for k, v in frozen.items())
               for s in [*profiles, *archives[baseline], *archives[treatment]])
    parity = {p: dict.fromkeys((*FIELDS, "corpus_membership"), 0) for p in policies}
    same_broad = 0
    same_request = dict.fromkeys((*FIELDS, "features"), 0)
    feature_max_abs = 0.0
    reuse_counts = {"available": 0, "encoded": 0, "questions_with_hits": 0}
    rows = []
    for shard in profiles:
        sample = shard["rows"]
        assert [r["id"] for r in sample] == ids[shard["shard"]::4][4:132:4]
        rng = Random(23847 + shard["shard"])
        for row in sample:
            order = list(policies)
            rng.shuffle(order)
            assert row["order"] == order
            for policy in policies:
                new, old = row["records"][policy], expected[policy][row["id"]]
                assert new["id"] == row["id"] and new["pool_hash"] == old["pool_hash"]
                assert set(new["arms"]) == {"broad", "conditioned_union"}
                phase = row["phase_ms"][policy]
                assert set(phase) == set(PHASES) and all(t > 0 for t in phase.values())
                assert sum(phase.values()) <= row["timing_ms"][policy]
                new_hits, old_hits = (r["corpus_hits"]["conditioned_union"] for r in (new, old))
                assert all(len(hits) == len(set(hits)) == 64 for hits in (new_hits, old_hits))
                parity[policy]["corpus_membership"] += set(new_hits) == set(old_hits)
                for k in FIELDS:
                    parity[policy][k] += new[k] == old[k]
            a, b = (row["records"][p] for p in policies)
            assert b["first"] == b["arms"]["broad"]["pair"][0]
            same_broad += a["arms"]["broad"] == b["arms"]["broad"]
            if reuse:
                ac, bc = a["sentence_counts"], b["sentence_counts"]
                assert ac["requested"] == ac["encoded"] == bc["requested"] >= bc["encoded"] > 0
                reuse_counts["available"] += bc["requested"]
                reuse_counts["encoded"] += bc["encoded"]
                reuse_counts["questions_with_hits"] += bc["encoded"] < bc["requested"]
                for k in same_request:
                    same_request[k] += a[k] == b[k]
                feature_max_abs = max(feature_max_abs,
                                      max(abs(x - y) for x, y in zip(a["features"], b["features"], strict=True)))
        rows.extend(sample)

    def times(subset):
        values = [{"serving_timing_ms": {"reference": r["timing_ms"][baseline],
                                         "shared": r["timing_ms"][treatment]}} for r in subset]
        t = metrics(values)
        return {"questions": len(subset), baseline: t["reference"], treatment: t["shared"],
                "paired_median_overhead_ms": -t["paired_median_saving_ms"],
                "paired_median_overhead_ci95_ms": [-v for v in reversed(t["paired_median_saving_ci95_ms"])],
                treatment + "_faster": t["shared_faster"]}

    return {**frozen, "reference_sha256": {p: sha256(refs[p].read_bytes()).hexdigest() for p in policies},
            "requests_per_policy": len(rows), "archived_parity": parity, "same_original_broad": same_broad,
            "prewarm_seconds": [round(s["prewarm_seconds"], 6) for s in profiles],
            "database_bytes": profiles[0]["database_bytes"], "all": times(rows),
            "by_order": {p + "_first": times([r for r in rows if r["order"][0] == p]) for p in policies},
            "phase_medians_ms": {p: {k: round(median(r["phase_ms"][p][k] for r in rows), 6) for k in PHASES}
                                 for p in policies},
            **({"same_request": same_request, "feature_max_abs_difference": round(feature_max_abs, 8),
                "sentence_encoding": reuse_counts} if reuse else {})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", nargs=4, type=Path)
    for name in ("fixture", "broad", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    method = p.add_mutually_exclusive_group(required=True)
    method.add_argument("--affinity", type=Path)
    method.add_argument("--sentence-reuse", action="store_true")
    a = p.parse_args()
    refs = {"live_broad": a.broad, "live_reuse": a.broad} if a.sentence_reuse else {
        "affinity": a.affinity, "live_broad": a.broad}
    result = evaluate([unpack(path) for path in a.profiles], a.fixture, refs)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
