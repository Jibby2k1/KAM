#!/usr/bin/env python3
"""Exact BM25 for the sealed BRIGHT queries and passage index."""

import argparse
from array import array
from collections import Counter
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from audit_kam_natural_data import digest, words

TOP = 128


def corpus(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def run(root):
    start = perf_counter()
    fixture = root / "fixture.json"
    bundle = json.loads(fixture.read_text())
    cases = bundle["rows"]
    query_terms = [sorted(set(words(q["question"]))) for q in cases]
    active = {w for terms in query_terms for w in terms}
    postings = {w: (array("I"), array("I")) for w in active}
    n = bundle["documents"]
    lengths = np.empty(n, dtype=np.int32)
    for i, part in enumerate(bundle["parts"]):
        filename = root / f"corpus-{i}.jsonl.gz"
        assert digest(filename) == part["sha256"]
        documents = list(corpus(filename))
        assert len(documents) == part["high"]-part["low"]
        for j, doc in enumerate(documents, part["low"]):
            token_counts = Counter(words(doc["content"]))
            lengths[j] = sum(token_counts.values())
            for x, count in token_counts.items():
                if x in postings:
                    where, counts = postings[x]
                    where.append(j)
                    counts.append(count)
    scale = 1.2 * (.25 + .75 * lengths / max(float(np.mean(lengths)), 1))
    saved = perf_counter()
    output = []
    for q, terms in zip(cases, query_terms, strict=True):
        time0 = perf_counter()
        scores = np.zeros(n, dtype=np.float64)
        for term in terms:
            where, counts = postings[term]
            if counts:
                loc, tf = np.asarray(where), np.asarray(counts)
                idf = np.log1p((n-len(loc)+.5)/(len(loc)+.5))
                scores[loc] += idf * tf * 2.2 / (tf + scale[loc])
        scores[q["excluded"]] = -np.inf
        ids = np.argsort(-scores, kind="stable")[:TOP].tolist()
        assert len(ids) == len(set(ids)) == TOP and not set(ids).intersection(q["excluded"])
        output.append({"id": q["id"], "ids": ids, "query_ms": 1000*(perf_counter()-time0)})
    return {"task": bundle["task"], "fixture_sha256": digest(fixture), "documents": n,
            "kernel": "BM25 k1=1.2 b=0.75 normalized words", "build_seconds": saved-start, "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    trial = run(a.inputs)
    a.output.write_bytes(gzip.compress(json.dumps(trial, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
