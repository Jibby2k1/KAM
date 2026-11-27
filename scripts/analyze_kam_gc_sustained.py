#!/usr/bin/env python3
"""Audit the sustained cached serving replay across three GC readiness policies."""

import argparse
from collections import Counter
from hashlib import sha256
import gzip
import json
from pathlib import Path
from statistics import median, quantiles

from analyze_kam_gc_readiness import MODES
from analyze_kam_live_anchor_profile import PHASES
from analyze_kam_shared_serving import unpack

ARM = "live_reuse"
METADATA = ("fixture_sha256", "pools_sha256", "config_sha256", "index_sha256", "fitted_sha256",
            "heads_sha256", "ablations_sha256", "baseline_sha256", "models", "guard_columns", "split")


def archive_digest(row):
    decisions = {k: row[k] for k in ("pool_hash", "first", "arms")}
    decisions["corpus_hits"] = {k: sorted(v) for k, v in row["corpus_hits"].items()}
    return sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()


def tails(values):
    percentiles = quantiles(values, n=100)
    return {"p50_ms": round(median(values), 6), "p95_ms": round(percentiles[94], 6),
            "p99_ms": round(percentiles[98], 6)}


def worker(shard, expected_ids, expected):
    assert shard["sustained_gc"] and shard["gpu"] == "NVIDIA B200"
    memory = shard["resident_kib_by_request"]
    assert [n for n, _ in memory] == [*range(0, 252, 16), 252]
    assert all(rss > 0 for _, rss in memory)
    assert 0 < shard["rss_before_requests_kib"] <= shard["rss_after_requests_kib"]
    assert [r["id"] for r in shard["rows"]] == expected_ids
    assert len(expected_ids) == 252
    latencies, pauses, hashes = [], [], []
    generations = Counter()
    archive_matches = 0
    for i, row in enumerate(shard["rows"], 1):
        assert row["order"] == [ARM] and set(row["records"]) == {ARM}
        ms = row["timing_ms"][ARM]
        phase = row["phase_ms"][ARM]
        assert set(phase) == set(PHASES) and all(v > 0 for v in phase.values())
        assert ms >= sum(phase.values())
        latencies.append(ms)
        record = row["records"][ARM]
        hashes.append(record["exact_sha256"])
        archive_matches += record["decision_sha256"] == expected[row["id"]]
        assert sum(event["ms"] for event in record["gc_events"]) <= ms
        for event in record["gc_events"]:
            g = event["generation"]
            assert g in (0, 1, 2) and event["ms"] > 0 and event["collected"] >= 0
            generations[g] += 1
            if g == 2:
                pauses.append({"request": i, "pause_ms": round(event["ms"], 6), "request_ms": round(ms, 6)})
    info = {
        "shard": shard["shard"], "node_sha256": shard["node_sha256"],
        "archive_matches": archive_matches, "latency": tails(latencies),
        "generation_counts": dict(generations), "generation_2": pauses,
        "readiness_ms": round(shard["gc_readiness_seconds"] * 1000, 6),
        "frozen_objects": shard["gc_frozen_objects"],
        "peak_rss_kib": shard["rss_after_requests_kib"],
        "resident_start_kib": memory[0][1], "resident_end_kib": memory[-1][1],
        "resident_peak_sampled_kib": max(rss for _, rss in memory),
    }
    assert (info["frozen_objects"] > 0) == (shard["gc_readiness_mode"] == "freeze")
    return info, latencies, hashes


def summarize(profiles, fixture, archive, diagnostic):
    fixture_raw = gzip.decompress(fixture.read_bytes())
    ids = [r["id"] for r in json.loads(fixture_raw)["rows"]["test"]]
    reference = unpack(archive)
    reference_rows = {r["id"]: r for s in reference for r in s["rows"]}
    expected = {i: archive_digest(row) for i, row in reference_rows.items()}
    assert set(ids) == expected.keys() and len(ids) == 1024
    metadata = {k: reference[0][k] for k in METADATA}
    assert metadata["fixture_sha256"] == sha256(fixture_raw).hexdigest()
    assert all(all(s[k] == v for k, v in metadata.items()) for s in (*profiles, *reference, diagnostic))
    assert len({s["database_bytes"] for s in profiles}) == 1
    assert set(s["gc_readiness_mode"] for s in profiles) == set(MODES)
    summary, digests = {}, {}
    for mode in MODES:
        shards = sorted((s for s in profiles if s["gc_readiness_mode"] == mode), key=lambda s: s["shard"])
        assert [s["shard"] for s in shards] == list(range(4))
        results = [worker(s, ids[s["shard"]::4][4:], expected) for s in shards]
        outcomes = [outcome[0] for outcome in results]
        summary[mode] = {"workers": outcomes,
                         "latency": tails([v for _, times, _ in results for v in times]),
                         "archive_matches": sum(row["archive_matches"] for row in outcomes),
                         "generation_2_count": sum(len(row["generation_2"]) for row in outcomes)}
        digests[mode] = [value for _, _, hashes in results for value in hashes]
    assert all([s["node_sha256"] for s in summary[m]["workers"]]
               == [s["node_sha256"] for s in summary["normal"]["workers"]] for m in MODES)
    comparisons = {}
    for baseline, treatment in (("normal", "collect"), ("collect", "freeze")):
        before = summary[baseline]["workers"]
        after = summary[treatment]["workers"]
        comparisons[f"{treatment}_vs_{baseline}"] = {
            "equal_full_decisions": sum(a == b for a, b in zip(digests[baseline], digests[treatment], strict=True)),
            "worker_p99_saving_ms": [round(x["latency"]["p99_ms"] - y["latency"]["p99_ms"], 6)
                                     for x, y in zip(before, after, strict=True)],
            "worker_peak_rss_change_kib": [y["peak_rss_kib"] - x["peak_rss_kib"]
                                          for x, y in zip(before, after, strict=True)],
        }
    [check] = diagnostic["rows"]
    live = check["records"][ARM]
    old = reference_rows[check["id"]]
    differences = []
    for mode in MODES:
        rows = [row for profile in profiles if profile["gc_readiness_mode"] == mode for row in profile["rows"]]
        [mismatch] = [r for r in rows if r["records"][ARM]["decision_sha256"] != expected[r["id"]]]
        assert check["id"] == mismatch["id"]
        assert archive_digest(live) == mismatch["records"][ARM]["decision_sha256"]
    assert all(live[k] == old[k] for k in ("pool_hash", "first", "arms"))
    assert live["corpus_hits"].keys() == old["corpus_hits"].keys()
    for arm, hits in live["corpus_hits"].items():
        old_hits = old["corpus_hits"][arm]
        if set(hits) != set(old_hits):
            differences.append({"arm": arm, "live_only": sorted(set(hits) - set(old_hits)),
                                "archive_only": sorted(set(old_hits) - set(hits))})
    return {"requests_per_mode": 1008, "reference_sha256": sha256(archive.read_bytes()).hexdigest(),
            "modes": summary, "comparisons": comparisons, "archive_mismatch": {
                "id": check["id"], "same_original_pool_anchor_and_final_outputs": True,
                "membership_differences": differences,
            }}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", nargs=12, type=Path)
    for name in ("fixture", "reference", "diagnostic", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    result = summarize([unpack(f) for f in args.profiles], args.fixture, args.reference, unpack(args.diagnostic))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
