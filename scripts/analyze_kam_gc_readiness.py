#!/usr/bin/env python3
"""Audit fresh worker processes with normal, collected and frozen Python state."""

import argparse
import json
from pathlib import Path

from analyze_kam_live_gc_trace import POLICIES, analyze
from analyze_kam_shared_serving import metrics, unpack

MODES = ("normal", "collect", "freeze")
FIDELITY = ("pool_hash", "first", "corpus_hits", "arms", "features", "sentence_counts")


def examine(profiles, fixture, reference):
    groups = {mode: sorted((s for s in profiles if s["gc_readiness_mode"] == mode),
                           key=lambda s: s["shard"]) for mode in MODES}
    assert len(profiles) == sum(map(len, groups.values())) == 12
    experiments = {mode: analyze(shards, fixture, reference) for mode, shards in groups.items()}
    readiness = {}
    records = {}
    for mode, shards in groups.items():
        assert [s["shard"] for s in shards] == list(range(4))
        for s in shards:
            assert s["gc_readiness_seconds"] >= 0 and s["gc_readiness_collected"] >= 0
            assert s["gc_frozen_objects"] > 0 if mode == "freeze" else s["gc_frozen_objects"] == 0
            assert 0 < s["rss_before_requests_kib"] <= s["rss_after_requests_kib"]
        readiness[mode] = [{k: round(s[k], 8) if isinstance(s[k], float) else s[k]
                            for k in ("shard", "gc_readiness_seconds", "gc_readiness_collected", "gc_frozen_objects",
                                      "rss_before_requests_kib", "rss_after_requests_kib")} for s in shards]
        records[mode] = {(s["shard"], r["id"]): r for s in shards for r in s["rows"]}
    ids = list(records["normal"])
    assert len(ids) == 128 and all(list(r) == ids for r in records.values())
    comparisons = {}
    for baseline, treatment in (("normal", "collect"), ("normal", "freeze"), ("collect", "freeze")):
        prev, new = records[baseline], records[treatment]
        assert all(prev[i]["order"] == new[i]["order"] for i in ids)
        comparisons[treatment + "_vs_" + baseline] = {
            arm: {"request": metrics([{"serving_timing_ms": {"reference": prev[i]["timing_ms"][arm],
                                                             "shared": new[i]["timing_ms"][arm]}} for i in ids]),
                  "parity": {k: sum(prev[i]["records"][arm][k] == new[i]["records"][arm][k] for i in ids)
                             for k in FIDELITY}} for arm in POLICIES}
    return {"readiness": readiness, "comparisons": comparisons, "independent_processes": experiments}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", nargs=12, type=Path)
    for name in ("fixture", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    payload = examine([unpack(f) for f in a.profiles], a.fixture, a.reference)
    a.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
