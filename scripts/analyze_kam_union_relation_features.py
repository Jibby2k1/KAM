#!/usr/bin/env python3
"""Recompute timing summaries and audit the cohort of an exact-feature microbenchmark."""

import argparse
from hashlib import sha256
import gzip
import json
from pathlib import Path
from statistics import median, quantiles


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def evaluate(profiles, fixture, pools, replay):
    ids = {q["id"] for q in json.loads(gzip.decompress(fixture.read_bytes()))["rows"]["test"]}
    original = json.loads(gzip.decompress(pools.read_bytes()))["rows"]
    audit = {r["id"]: len(set(original[r["id"]]) | set(r["corpus_hits"]["conditioned_union"]))
             for shard in json.loads(gzip.decompress(replay.read_bytes())) for r in shard["rows"]}
    assert ids == original.keys() == audit.keys()
    seen = set()
    fields = {k + "_sha256": digest(v) for k, v in (("fixture", fixture), ("pools", pools), ("replay", replay))}
    outputs = []
    for profile in profiles:
        assert all(profile[key] == value for key, value in fields.items())
        assert profile["cpu_threads"] == 1
        rows = profile["rows"]
        assert profile["bitwise_parity"] == len(rows)
        for row in rows:
            key = row["id"], row["pool"]
            assert key not in seen and key[0] in ids
            assert key[1] in ("original", "conditioned_union")
            assert row["documents"] == (64 if key[1] == "original" else audit[key[0]])
            seen.add(key)
        blocks = {}
        for pool, ref in profile["summary"].items():
            selected = [r for r in rows if r["pool"] == pool]
            summary = {}
            for method in ("baseline_ms", "bitset_ms"):
                values = [r[method] for r in selected]
                assert all(v > 0 for v in values)
                summary[method] = {"p50_ms": round(median(values), 6),
                                   "p95_ms": round(quantiles(values, n=20)[18], 6)}
            assert summary == ref
            blocks[pool] = {"n": len(selected), **summary,
                            "median_speedup": round(summary["baseline_ms"]["p50_ms"] /
                                                    summary["bitset_ms"]["p50_ms"], 3)}
        assert blocks.keys() == {"original", "conditioned_union"}
        outputs.append({"cpu_model": profile["cpu_model"], "torch_version": profile["torch_version"],
                        "cpu_threads": profile["cpu_threads"], "bitwise_parity": len(rows), "pools": blocks})
    assert len(seen) == 2 * len(ids)
    return {**fields, "questions": len(ids), "bitwise_parity": len(seen), "profiles": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path)
    for name in ("fixture", "pools", "replay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    result = evaluate([load(p) for p in a.profiles], a.fixture, a.pools, a.replay)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
