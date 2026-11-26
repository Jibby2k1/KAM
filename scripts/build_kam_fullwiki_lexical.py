#!/usr/bin/env python3
"""Build one global BM25 index and an ID-aligned article store."""

import argparse
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
import tantivy

from audit_kam_natural_data import digest
from prepare_kam_natural_router import title_key


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(a):
    start = perf_counter()
    meta = json.loads((a.inputs / "manifest.json").read_text())
    banned_path, qpath = a.inputs / "excluded_titles.txt", a.inputs / "queries.json"
    assert digest(banned_path) == meta["excluded_titles_sha256"] and digest(qpath) == meta["queries_sha256"]
    seen = set(banned_path.read_text().splitlines())
    assert len(seen) == meta["excluded_titles"]
    queries = json.loads(qpath.read_text())
    protected = {x for rows in queries.values() for row in rows for x in row["gold_titles"]}
    assert not protected & seen
    source = pq.ParquetFile(a.inputs / "corpus.parquet")
    assert source.metadata.num_rows == meta["total_rows"]

    # Publish the immutable artifacts only after both stores and the dense ID mask are complete.
    work = a.output
    work.mkdir(exist_ok=False, parents=True)
    search_path = work / "tantivy"
    search_path.mkdir()
    builder = tantivy.SchemaBuilder()
    builder.add_integer_field("row_id", stored=True, fast=a.fast_row_id)
    builder.add_text_field("contents", index_option="freq")
    schema = builder.build()
    index = tantivy.Index(schema, path=str(search_path))
    writer = index.writer(heap_size=768 * 1024 * 1024, num_threads=a.threads)
    connection = sqlite3.connect(work / "articles.sqlite")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL)")
    mask = np.zeros(meta["total_rows"], dtype=np.bool_)
    raw_id = 0
    kept = 0
    matched = set()
    cap = min(a.limit, len(mask)) if a.limit is not None else len(mask)
    for i, batch in enumerate(source.iter_batches(batch_size=16384, columns=["title", "text"])):
        if raw_id >= cap:
            break
        contents = []
        for row in batch.slice(0, cap - raw_id).to_pylist():
            row_id, raw_id = raw_id, raw_id + 1
            title, body = row["title"] or "", row["text"] or ""
            normalized = title_key(title)
            if not normalized or normalized in seen or not body.strip():
                continue
            seen.add(normalized)
            if normalized in protected:
                matched.add(normalized)
            mask[row_id] = True
            writer.add_document(tantivy.Document.from_dict({"row_id": row_id, "contents": title + ". " + body}, schema))
            contents.append((row_id, title, body))
        connection.executemany("INSERT INTO docs(id,title,body) VALUES (?,?,?)", contents)
        kept += len(contents)
        if i % 32 == 31:
            print(f"Indexed {kept}/{raw_id} rows, elapsed {perf_counter()-start:.1f}s", flush=True)
    connection.commit()
    connection.close()
    writer.commit()
    writer.wait_merging_threads()
    index.reload()
    assert raw_id == cap and mask.sum() == kept == index.searcher().num_docs
    np.save(work / "allowed.npy", mask, allow_pickle=False)
    stats = {
        "source_sha256": meta["source_sha256"], "queries_sha256": meta["queries_sha256"],
        "excluded_titles_sha256": meta["excluded_titles_sha256"], "scanned": raw_id, "indexed": kept,
        "tantivy_version": tantivy.__version__, "threads": a.threads,
        "fast_row_id": a.fast_row_id,
        "seconds": perf_counter() - start, "mask_sha256": digest(work / "allowed.npy"),
        "storage_bytes": sum(p.stat().st_size for p in work.rglob("*") if p.is_file()),
        "coverage": {g: {"n": len(queries[g]), "both_titles": sum(set(r["gold_titles"]) <= matched
                                                                 for r in queries[g])} for g in queries},
    }
    write(work / "completed.json", stats)
    print(json.dumps(stats), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--fast-row-id", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
