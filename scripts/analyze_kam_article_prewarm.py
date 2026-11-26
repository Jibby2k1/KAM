#!/usr/bin/env python3
"""Audit crossover assignments and summarize the first-touch prewarm experiment."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import median, quantiles


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def evaluate(records, pools, reference):
    pools_hash, reference_hash = (sha256(p.read_bytes()).hexdigest() for p in (pools, reference))
    assert [r["rank"] for r in records] == [0, 1]
    assert len({r["node_sha256"] for r in records}) == 2
    assert len({r["database_bytes"] for r in records}) == 1
    pools, reference = read(pools)["rows"], read(reference)
    case_sizes = {r["id"]: len(set(pools[r["id"]]) | set(r["corpus_hits"]["conditioned_union"]))
                  for s in reference for r in s["rows"]}
    assert set(case_sizes) == pools.keys() and len(pools) == 1024
    chosen = Random(20261127).sample(sorted(pools), 256)

    def summarize(rows):
        times = [r["latency_ms"] for r in rows]
        assert all(t > 0 for t in times)
        return {"p50_ms": round(median(times), 6), "p95_ms": round(quantiles(times, n=20)[18], 6)}

    by_node = []
    cold, warm = {}, {}
    for r in records:
        assert r["pools_sha256"] == pools_hash and r["reference_sha256"] == reference_hash
        before, after = r["baseline"], r["prewarmed"]
        assert [q["id"] for q in before] == chosen[r["rank"]::2]
        assert [q["id"] for q in after] == chosen[1 - r["rank"]::2]
        assert all(q["articles"] == case_sizes[q["id"]] for q in (*before, *after))
        assert summarize(before) == r["baseline_stats"] and summarize(after) == r["prewarmed_stats"]
        assert r["prewarm_seconds"] > 0
        cold.update((q["id"], q) for q in before)
        warm.update((q["id"], q) for q in after)
        by_node.append({k: r[k] for k in ("rank", "prewarm_seconds", "baseline_stats", "prewarmed_stats")})
    assert set(chosen) == cold.keys() == warm.keys()
    assert all(cold[i]["ordered_text_sha256"] == warm[i]["ordered_text_sha256"] for i in chosen)
    savings = [cold[i]["latency_ms"] - warm[i]["latency_ms"] for i in chosen]
    rand = Random(5801)
    intervals = quantiles([median(rand.choices(savings, k=len(savings))) for _ in range(5000)], n=40)
    return {"pools_sha256": pools_hash, "reference_sha256": reference_hash, "requests": len(chosen),
            "database_bytes": records[0]["database_bytes"], "by_node": by_node,
            "baseline": summarize(list(cold.values())), "prewarmed": summarize(list(warm.values())),
            "exact_text_parity": len(chosen), "median_paired_saving_ms": round(median(savings), 6),
            "median_paired_saving_ci95_ms": [round(intervals[0], 6), round(intervals[-1], 6)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", nargs=2, type=Path)
    for name in ("pools", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = evaluate([json.loads(h.read_text()) for h in a.profiles], a.pools, a.reference)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
