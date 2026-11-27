#!/usr/bin/env python3
"""Freeze held-out MultiHop-RAG questions with both fact spans present in source text."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from audit_kam_natural_data import digest, stable, words

KINDS = ("comparison_query", "temporal_query", "inference_query")
QUOTA = 128


def norm(text):
    return " ".join(words(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "corpus", "previous", "fixture", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    article = json.loads(a.corpus.read_text())
    body = {r["url"]: norm(r["body"]) for r in article}
    ids = {r["url"]: i for i, r in enumerate(article)}
    assert len(body) == len(article) == 609
    prior = json.loads(a.previous.read_text())
    corpus = digest(a.corpus)
    assert prior["corpus_sha256"] == corpus
    reserve = {norm(q["question"]) for q in prior["rows"]}
    used = set(reserve)
    counts = Counter()
    cases = []
    for q in sorted(json.loads(a.source.read_text()), key=lambda q: stable("news-windows-v1:" + q["query"])):
        text = norm(q["query"])
        kind = q["question_type"]
        source = q["evidence_list"]
        urls = {e["url"] for e in source}
        if text in used or len(urls) != 2 or not urls <= body.keys() or kind not in KINDS or counts[kind] >= QUOTA:
            continue
        if any(not norm(e["fact"]) or norm(e["fact"]) not in body[e["url"]] for e in source):
            continue
        assert isinstance(q["answer"], str)
        evidence = {ids[url]: list(dict.fromkeys(e["fact"] for e in source if e["url"] == url)) for url in sorted(urls)}
        assert len(evidence) == 2
        cases.append({"id": stable(q["query"])[:20], "kind": kind, "question": q["query"], "answer": q["answer"],
                      "evidence": evidence})
        used.add(text)
        counts[kind] += 1
    assert counts == {k: QUOTA for k in KINDS}
    payload = {"source_sha256": digest(a.source), "corpus_sha256": corpus, "previous_sha256": digest(a.previous),
               "previous_questions": len(reserve), "counts": counts, "rows": cases}
    a.fixture.write_bytes(gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0))
    public = payload | {"fixture_sha256": digest(a.fixture),
                        "rows": [{"id": q["id"], "kind": q["kind"], "article_ids": list(q["evidence"])} for q in cases]}
    a.manifest.write_bytes(gzip.compress(json.dumps(public, separators=(",", ":")).encode(), mtime=0))
    print(counts)


if __name__ == "__main__":
    main()
