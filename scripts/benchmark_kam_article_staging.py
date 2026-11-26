#!/usr/bin/env python3
"""Measure exact article reads before and after staging the SQLite index locally."""

import argparse
from contextlib import closing
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sqlite3
from statistics import median, quantiles
from tempfile import TemporaryDirectory
from time import perf_counter


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def file_digest(path):
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def connection(path):
    db = sqlite3.connect(f"{path.resolve().as_uri()}?immutable=1", uri=True)
    db.execute("PRAGMA mmap_size=1073741824")
    return db


def batch_fetch(db, ids):
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = {i: (title, body) for i, title, body in db.execute(
        f"SELECT id,title,body FROM docs WHERE id IN ({placeholders})", ids)}
    assert len(rows) == len(ids)
    return [rows[i] for i in ids]


def fetch_request(db, original, added):
    t = perf_counter()
    values = batch_fetch(db, original) + batch_fetch(db, added)
    latency = (perf_counter() - t) * 1000
    fingerprint = sha256(json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return latency, fingerprint


def percentile(values):
    return {"p50_ms": round(median(values), 6), "p95_ms": round(quantiles(values, n=20)[18], 6)}


def benchmark(a):
    sealed = unpack(a.pools)
    refs = unpack(a.reference)
    assert all(s["pools_sha256"] == file_digest(a.pools) for s in refs)
    ordered = sorted((r for shard in refs for r in shard["rows"]), key=lambda r: r["id"])
    cases = []
    for row in ordered[a.offset::a.stride]:
        original = sealed["rows"][row["id"]]
        assert row["pool_hash"] == sha256(json.dumps(original, separators=(",", ":")).encode()).hexdigest()
        base = set(original)
        added = [i for i in row["corpus_hits"]["conditioned_union"] if i not in base]
        assert len(original) == len(base) == 64 and len(added) == len(set(added))
        cases.append((row["id"], original, added))
    assert len(cases) == 128
    bytes_ = a.database.stat().st_size
    assert shutil.disk_usage(a.scratch).free > 2 * bytes_

    # Before copying the database, take the original workload's first-touch measurement.
    with closing(connection(a.database)) as remote:
        first = [fetch_request(remote, p, h) for _, p, h in cases]

    with TemporaryDirectory(prefix="kam-articles-", dir=a.scratch) as directory:
        local_path = Path(directory) / "articles.sqlite"
        start = perf_counter()
        shutil.copyfile(a.database, local_path)
        stage_s = perf_counter() - start
        assert local_path.stat().st_size == bytes_
        results = []
        with closing(connection(a.database)) as remote, closing(connection(local_path)) as local:
            paths = {"shared_after_staging": remote, "local_after_staging": local}
            for i, ((id_, p, h), (first_ms, original_digest)) in enumerate(zip(cases, first, strict=True)):
                order = tuple(paths) if i % 2 == 0 else tuple(reversed(paths))
                row = {"id": id_, "shared_first_touch": first_ms, "articles": len(p) + len(h)}
                for name in order:
                    elapsed, text_digest = fetch_request(paths[name], p, h)
                    assert text_digest == original_digest, (id_, name)
                    row[name] = elapsed
                results.append(row)
        sha = file_digest(local_path)
        assert sha == file_digest(a.database)
        same_device = os.stat(local_path).st_dev == os.stat(a.database).st_dev
    methods = ("shared_first_touch", "shared_after_staging", "local_after_staging")
    return {"database_bytes": bytes_, "database_sha256": sha, "copy_seconds": stage_s,
            "same_filesystem_device": same_device, "pools_sha256": file_digest(a.pools),
            "reference_sha256": file_digest(a.reference), "offset": a.offset, "stride": a.stride,
            "exact_text_parity": len(results), "summary": {k: percentile([r[k] for r in results]) for k in methods},
            "rows": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pools", "reference", "database", "scratch", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--stride", type=int, default=8)
    a = parser.parse_args()
    assert 0 <= a.offset < a.stride
    a.output.write_text(json.dumps(benchmark(a), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
