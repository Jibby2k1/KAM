#!/usr/bin/env python3
"""Audit end-to-end paired outputs and summarize CPU relation feature cost."""

import argparse
from collections import Counter
from hashlib import sha256
import gzip
import json
from pathlib import Path
from statistics import median, quantiles


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def evaluate(reference, measured, n):
    assert len(reference) == len(measured) == 4
    assert sorted(s["shard"] for s in reference) == sorted(s["shard"] for s in measured) == list(range(4))
    keys = ("fixture_sha256", "pools_sha256", "config_sha256", "index_sha256", "fitted_sha256", "heads_sha256",
            "ablations_sha256", "baseline_sha256", "models", "indexed_documents", "guard_columns", "split")
    provenance = {k: reference[0][k] for k in keys}
    assert all(all(shard[k] == v for k, v in provenance.items()) for shard in (*reference, *measured))

    old = {r["id"]: r for shard in reference for r in shard["rows"]}
    rows = [r for shard in measured for r in shard["rows"]]
    assert len(rows) == len(old) == n
    assert {r["id"] for r in rows} == old.keys()
    row_keys = ("pool_hash", "first", "corpus_hits", "features")
    arm_keys = ("pair", "gold_pair_in_shortlist", "exact")
    counts = Counter(dict.fromkeys((*row_keys, *arm_keys), 0))
    for row in rows:
        before = old[row["id"]]
        for key in row_keys:
            counts[key] += before[key] == row[key]
        assert row["arms"].keys() == before["arms"].keys() == {"broad", "conditioned_union"}
        for key in arm_keys:
            counts[key] += all(arm[key] == before["arms"][name][key] for name, arm in row["arms"].items())
        assert set(row["relation_timing_ms"]) == {"reference", "shared"}
        assert all(v > 0 for v in row["relation_timing_ms"].values())

    timings = {}
    for method in ("reference", "shared"):
        values = [r["relation_timing_ms"][method] for r in rows]
        timings[method] = {"p50_ms": round(median(values), 6),
                           "p95_ms": round(quantiles(values, n=20)[18], 6)}
    return {"questions": n, "matching_rows": dict(counts), "gpus": dict(Counter(s["gpu"] for s in measured)),
            "timing": timings, "median_speedup": round(timings["reference"]["p50_ms"] /
                                                      timings["shared"]["p50_ms"], 3),
            "provenance": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-replay", type=Path)
    a = parser.parse_args()
    measured = ([unpack(p) for p in sorted(a.input.glob("test-*.json.gz"))] if a.input.is_dir() else unpack(a.input))
    summary = evaluate(unpack(a.reference), measured, 1024)
    summary["reference_sha256"] = sha256(a.reference.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        a.write_replay.write_bytes(gzip.compress(json.dumps(measured, separators=(",", ":"), allow_nan=False).encode(),
                                                 mtime=0))


if __name__ == "__main__":
    main()
