#!/usr/bin/env python3
"""Measure whether a frozen Wikipedia lead index contains cross-dataset evidence."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import unicodedata

from audit_kam_natural_data import digest, words
from prepare_kam_natural_router import title_key


def tokens(text):
    return words(unicodedata.normalize("NFKC", text))


def twowiki(source, rejected):
    import pyarrow.parquet as pq

    for row in pq.read_table(source, columns=["id", "type", "context", "supporting_facts"]).to_pylist():
        facts = row["supporting_facts"]
        context = row["context"]
        titles = dict(zip(context["title"], context["sentences"], strict=True))
        attachments = {}
        for title, i in zip(facts["title"], facts["sent_id"], strict=True):
            attachments[title, i] = titles[title][i] if title in titles and 0 <= i < len(titles[title]) else None
        if len({title_key(t) for t, _ in attachments}) < 2 or not all(attachments.values()):
            rejected["invalid_support"] += 1
            continue
        yield {"id": row["id"], "type": row["type"], "units": [(t, text) for (t, _), text in attachments.items()]}


def musique(source, rejected):
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row["answerable"]:
                rejected["unanswerable"] += 1
                continue
            support = [(p["title"], p["paragraph_text"]) for p in row["paragraphs"] if p["is_supporting"]]
            if not support or not all(t and s for t, s in support):
                rejected["invalid_support"] += 1
                continue
            yield {"id": row["id"], "type": row["id"].split("__")[0], "units": support}


def article_text(database, targets, expected_n):
    found = {}
    with sqlite3.connect(f"{database.as_uri()}?immutable=1", uri=True) as db:
        assert db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == expected_n
        for i, title in db.execute("SELECT id, title FROM docs"):
            normalized = title_key(title)
            if normalized in targets:
                assert normalized not in found
                found[normalized] = i
        result = {}
        for title, i in sorted(found.items()):
            text, = db.execute("SELECT body FROM docs WHERE id = ?", (i,)).fetchone()
            result[title] = (i, tokens(text))
    return result


def wordspan(body, passage):
    # Surround strings so a partial token cannot count as exact evidence.
    clean = f" {' '.join(body)} "
    span = f" {' '.join(passage)} "
    width = min(12, len(passage))
    exact = span in clean
    partial = exact or any(f" {' '.join(passage[i:i+width])} " in clean for i in range(len(passage) - width + 1))
    return exact, partial


def audit(inputs, sources):
    lexical = inputs / "lexical"
    meta = json.loads((lexical / "completed.json").read_text())
    omitted = set((inputs / "excluded_titles.txt").read_text().splitlines())
    assert meta["excluded_titles_sha256"] == digest(inputs / "excluded_titles.txt")
    loaders = {"twowiki": twowiki, "musique": musique}
    cohorts, rejections = {}, {}
    for k, path in sources.items():
        counts = Counter()
        cohorts[k] = list(loaders[k](path, counts))
        rejections[k] = dict(counts)
        assert len(cohorts[k]) == len({r["id"] for r in cohorts[k]})
    targets = {title_key(t) for cases in cohorts.values() for row in cases for t, _ in row["units"]}
    found = article_text((lexical / "articles.sqlite").resolve(), targets, meta["indexed"])
    assert not found.keys() & omitted
    output = {}
    for dataset, rows in cohorts.items():
        expected = []
        for row in rows:
            units = []
            for title, text in row["units"]:
                key = title_key(title)
                query = tokens(text)
                assert key and query
                record = {"title_sha256": sha256(key.encode()).hexdigest(), "indexed": key in found,
                          "excluded": key in omitted, "words": len(query)}
                if key in found:
                    i, body = found[key]
                    exact, partial = wordspan(body, query)
                    record.update({"corpus_id": i, "body_words": len(body), "exact_unit": exact, "shared_excerpt": partial})
                units.append(record)
            expected.append({"id": row["id"], "type": row["type"], "units": units})
        output[dataset] = expected
    return {"corpus": {k: meta[k] for k in ("source_sha256", "mask_sha256", "excluded_titles_sha256", "indexed")},
            "data_sha256": {k: digest(path) for k, path in sources.items()}, "excluded_source_records": rejections,
            "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("inputs", "twowiki", "musique", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    a = p.parse_args()
    result = audit(a.inputs, {"twowiki": a.twowiki, "musique": a.musique})
    a.output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
