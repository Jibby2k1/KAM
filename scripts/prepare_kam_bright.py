#!/usr/bin/env python3
"""Seal two natural BRIGHT domains and validate gold corpora without reading rationales."""

import argparse
import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq

from audit_kam_natural_data import digest

REVISION = "8c3fa985016abc6b0e914d82dd01950e0669c0b7"
DOMAINS = ("economics", "earth_science")
REPLICATION_DOMAINS = ("psychology", "sustainable_living")
SHARDS = 4
NONEXCLUSION = "N/A"


def encode(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def prepare(source, output_dir, domain):
    rows_path, corpus_path = [source / f"{which}-{domain}.parquet" for which in ("examples", "documents")]
    store = output_dir / domain
    store.mkdir(parents=True, exist_ok=True)
    documents = pq.read_table(corpus_path, columns=["id", "content"]).to_pylist()
    queries = pq.read_table(rows_path, columns=["id", "query", "excluded_ids", "gold_ids"]).to_pylist()
    lookup = {r["id"]: i for i, r in enumerate(documents)}
    assert len(lookup) == len(documents) and all(isinstance(r["content"], str) and r["content"].strip() for r in documents)
    assert len(queries) == len({q["id"] for q in queries})
    frozen = []
    exclusions_missing = corpus_missing = 0
    for q in queries:
        assert q["query"].strip() and q["gold_ids"]
        gold = sorted({lookup[d] for d in q["gold_ids"] if d in lookup})
        banned = sorted({lookup[d] for d in q["excluded_ids"] if d in lookup})
        absent = sorted(set(q["gold_ids"]).difference(lookup))
        missing = {d for d in q["excluded_ids"] if d != NONEXCLUSION and d not in lookup}
        assert not set(gold).intersection(banned)
        frozen.append({"id": q["id"], "question": q["query"], "gold": gold, "excluded": banned,
                       "unindexed_gold": len(absent), "unindexed_exclusions": len(missing),
                       "gold_docs": sorted(set(q["gold_ids"]))})
        corpus_missing += len(absent)
        exclusions_missing += len(missing)
    parts = []
    for i in range(SHARDS):
        low, high = [(j * len(documents)) // SHARDS for j in (i, i+1)]
        current = documents[low:high]
        path = store / f"corpus-{i}.jsonl.gz"
        with path.open("wb") as file, gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as stream:
            for d in current:
                stream.write(encode(d) + b"\n")
        parts.append({"low": low, "high": high, "sha256": digest(path)})
    provenance = {"source_revision": REVISION, "task": domain, "documents": len(documents), "queries": len(queries),
                  "source_digests": {"examples": digest(rows_path), "docs": digest(corpus_path)},
                  "nonindexed_gold": corpus_missing, "nonindexed_exclusions": exclusions_missing, "parts": parts}
    fixture = store / "fixture.json"
    fixture.write_bytes(encode(provenance | {"rows": frozen}) + b"\n")
    public = [{k: v for k, v in row.items() if k != "question"} for row in frozen]
    (store / "manifest.json").write_bytes(encode(provenance | {"fixture_sha256": digest(fixture), "rows": public}) + b"\n")
    print(domain, len(queries), len(documents), corpus_missing, exclusions_missing)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--domains", nargs="+", choices=(*DOMAINS, *REPLICATION_DOMAINS), default=DOMAINS)
    a = p.parse_args()
    for domain in a.domains:
        prepare(a.sources, a.output, domain)


if __name__ == "__main__":
    main()
