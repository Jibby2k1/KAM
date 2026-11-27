#!/usr/bin/env python3
"""Cross first-pass and prewarmed exact lexical search on two HPG hosts."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter
from os import uname

import numpy as np
import tantivy

from audit_kam_natural_data import digest, words
from benchmark_kam_article_prewarm import barrier, read_index

DEPTH = 128


def sampling(fixture):
    queries = json.loads(gzip.decompress(fixture.read_bytes()))["rows"]["test"]
    assert len({q["id"] for q in queries}) == len(queries) == 1024
    sample = [q for shard in range(4) for q in queries[shard::4][4:132:4]]
    selected = {q["id"] for q in sample}
    assert len(sample) == len(selected) == 128
    shuffle = Random(382633)
    shuffle.shuffle(sample)
    return sample, [q for q in queries if q["id"] not in selected][:4]


def lexical(index, searcher, question, fallback):
    # Same parser, depth, document retrieval, and ordering as the live exact hybrid path.
    text = " ".join(f'"{token}"' for token in sorted(set(words(question))))
    query = index.parse_query(text, ["contents"])
    hits = searcher.search(query, limit=DEPTH, count=False).hits
    ids = [searcher.doc(address).get_first("row_id") for _, address in hits]
    remaining = set(ids)
    if len(ids) < DEPTH:
        ids.extend(i for i in fallback if i not in remaining and len(ids) < DEPTH)
    assert len(ids) == len(set(ids)) == DEPTH
    return ids


def run(a):
    n = a.rank
    region = a.lexical / "tantivy"
    serial = tantivy.Index.open(str(region))
    parallel = tantivy.Index.open(str(region), search_threads=2)
    routes = {k: (x, x.searcher()) for k, x in (("serial", serial), ("two_threads", parallel))}
    meta = json.loads((a.lexical / "completed.json").read_text())
    assert all(s.num_docs == meta["indexed"] for _, s in routes.values())
    mask = np.load(a.lexical / "allowed.npy", allow_pickle=False)
    fallback = np.flatnonzero(mask)[:2048].tolist()
    sample, warmup = sampling(a.fixture)

    def search(row, kind):
        before = perf_counter()
        candidates = lexical(*routes[kind], row["question"], fallback)
        return {"id": row["id"], "kind": kind, "ms": (perf_counter() - before) * 1000, "ordered": candidates}

    for row in warmup:
        search(row, "serial")
    baseline = [search(row, "serial") for row in sample[n::2]]
    barrier(a.working_dir, n)
    files = sorted(p for p in region.iterdir() if p.is_file() and not p.name.startswith("."))
    assert len(files) > 5
    bytes_, elapsed = zip(*(read_index(p) for p in files), strict=True)
    warmed = a.working_dir / "warmed"
    warmed.mkdir(exist_ok=True)
    barrier(warmed, n)
    warm = []
    rng = Random(82791 + n)
    for row in sample[1 - n::2]:
        order = rng.sample(list(routes), k=2)
        warm.append({"id": row["id"], "order": order, "routes": {kind: search(row, kind) for kind in order}})
    return {"rank": n, "node_sha256": sha256(uname().nodename.encode()).hexdigest(),
            "fixture_sha256": digest(a.fixture), "index_sha256": digest(a.lexical / "completed.json"),
            "read_bytes": sum(bytes_), "read_seconds": sum(elapsed), "read_files": len(files),
            "baseline": baseline, "prewarmed": warm}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "lexical", "working-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rank", type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
