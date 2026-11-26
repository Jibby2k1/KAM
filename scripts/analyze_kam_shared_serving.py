#!/usr/bin/env python3
"""Audit and summarize paired warm-request timings for shared relation features."""

import argparse
from collections import defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path
from random import Random
from statistics import median, quantiles


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def metrics(rows):
    t = {k: [r["serving_timing_ms"][k] for r in rows] for k in ("reference", "shared")}
    latency = {k: {"p50_ms": round(median(v), 6), "p95_ms": round(quantiles(v, n=20)[18], 6)}
               for k, v in t.items()}
    deltas = [a - b for a, b in zip(t["reference"], t["shared"], strict=True)]
    rng = Random(9207)
    boot = [median(rng.choices(deltas, k=len(deltas))) for _ in range(5000)]
    ci = quantiles(boot, n=40)
    return {"questions": len(rows), **latency, "median_speedup": round(
            latency["reference"]["p50_ms"] / latency["shared"]["p50_ms"], 6),
            "paired_median_saving_ms": round(median(deltas), 6),
            "paired_median_saving_ci95_ms": [round(ci[0], 6), round(ci[-1], 6)],
            "shared_faster": sum(d > 0 for d in deltas)}


def evaluate(shards, fixture_path, reference_path):
    reference = unpack(reference_path)
    expected = {r["id"]: r for s in reference for r in s["rows"]}
    fixture = gzip.decompress(fixture_path.read_bytes())
    ids = [row["id"] for row in json.loads(fixture)["rows"]["test"]]
    assert set(ids) == expected.keys() and len(ids) == 1024
    keys = ("fixture_sha256", "pools_sha256", "config_sha256", "index_sha256", "fitted_sha256", "heads_sha256",
            "ablations_sha256", "baseline_sha256", "models", "guard_columns", "split")
    provenance = {k: reference[0][k] for k in keys}
    assert provenance["fixture_sha256"] == sha256(fixture).hexdigest()
    assert all(all(s[k] == v for k, v in provenance.items()) for s in (*reference, *shards))
    assert len(shards) == 4 and sorted(s["shard"] for s in shards) == list(range(4))
    cache_modes = {s.get("serving_cache", "first_pass") for s in shards}
    assert len(cache_modes) == 1 and cache_modes <= {"first_pass", "primed"}
    gpus = defaultdict(list)
    archived_matches = 0
    for shard in shards:
        rows = shard["rows"]
        assert len(rows) == 32
        assert [r["id"] for r in rows] == ids[shard["shard"]::4][4:132:4]
        assert [r["reference_first"] for r in rows] == [i % 2 == 0 for i in range(32)]
        for row in rows:
            old = expected[row["id"]]
            assert row["pool_hash"] == old["pool_hash"]
            assert set(row["serving_timing_ms"]) == {"reference", "shared"}
            assert all(v > 0 for v in row["serving_timing_ms"].values())
            archived_matches += all(r["pair"] == old["arms"][k]["pair"] and r["exact"] == old["arms"][k]["exact"]
                                    for k, r in row["arms"].items())
        gpus[shard["gpu"]].extend(rows)
    rows = [r for s in shards for r in s["rows"]]
    order = {"reference_first": metrics([r for r in rows if r["reference_first"]]),
             "shared_first": metrics([r for r in rows if not r["reference_first"]])}
    return {**provenance, "reference_replay_sha256": sha256(reference_path.read_bytes()).hexdigest(),
            "serving_cache": cache_modes.pop(),
            "paired_evidence_and_labels": len(rows), "matches_archived_batched_evidence_and_labels": archived_matches,
            "all": metrics(rows), "by_gpu": {k: metrics(v) for k, v in sorted(gpus.items())}, "by_order": order}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    for name in ("fixture", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--write-replay", type=Path)
    a = p.parse_args()
    shards = ([unpack(p) for p in sorted(a.input.glob("serving-*.json.gz"))] if a.input.is_dir() else unpack(a.input))
    result = evaluate(shards, a.fixture, a.reference)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        a.write_replay.write_bytes(gzip.compress(json.dumps(shards, separators=(",", ":"), allow_nan=False).encode(),
                                                 mtime=0))


if __name__ == "__main__":
    main()
