#!/usr/bin/env python3
"""Test shared-index prewarming on disjoint requests with crossed node assignments."""

import argparse
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random
from time import monotonic, perf_counter, sleep

from benchmark_kam_article_staging import connection, fetch_request, file_digest, percentile, unpack


def read_index(path):
    started = perf_counter()
    bytes_read = 0
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            bytes_read += len(block)
    assert bytes_read == path.stat().st_size
    return bytes_read, perf_counter() - started


def barrier(path, rank):
    (path / f"baseline-{rank}").touch()
    deadline = monotonic() + 90
    while not (path / f"baseline-{1 - rank}").is_file():
        if monotonic() >= deadline:
            raise TimeoutError("the other CPU node did not reach the baseline barrier")
        sleep(.2)


def inputs(a):
    pools = unpack(a.pools)
    refs = unpack(a.reference)
    assert all(s["pools_sha256"] == file_digest(a.pools) for s in refs)
    rows = sorted((row for s in refs for row in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == len(pools["rows"]) == 1024
    chosen = Random(20261127).sample(rows, 256)
    cases = []
    for row in chosen:
        original = pools["rows"][row["id"]]
        base = set(original)
        assert len(original) == len(base) == 64
        assert row["pool_hash"] == sha256(json.dumps(original, separators=(",", ":")).encode()).hexdigest()
        extra = [doc for doc in row["corpus_hits"]["conditioned_union"] if doc not in base]
        assert len(extra) == len(set(extra))
        cases.append((row["id"], original, extra))
    return cases


def measure(db, cases):
    rows = []
    for id_, base, extra in cases:
        elapsed, text_digest = fetch_request(db, base, extra)
        rows.append({"id": id_, "articles": len(base) + len(extra),
                     "latency_ms": elapsed, "ordered_text_sha256": text_digest})
    return rows


def run(a):
    rank = a.rank
    cases = inputs(a)
    # Every measured question appears once per node. Warm/cold order at the node
    # is fixed, and assignments cross the two nodes to balance query difficulty.
    with closing(connection(a.database)) as db:
        baseline = measure(db, cases[rank::2])
    barrier(a.working_dir, rank)
    bytes_, seconds = read_index(a.database)
    with closing(connection(a.database)) as db:
        warm = measure(db, cases[1 - rank::2])
    return {"rank": rank, "node_sha256": sha256(os.uname().nodename.encode()).hexdigest(), "database_bytes": bytes_, "prewarm_seconds": seconds,
            "pools_sha256": file_digest(a.pools), "reference_sha256": file_digest(a.reference),
            "baseline_stats": percentile([r["latency_ms"] for r in baseline]),
            "prewarmed_stats": percentile([r["latency_ms"] for r in warm]), "baseline": baseline, "prewarmed": warm}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pools", "reference", "database", "working-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rank", type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
