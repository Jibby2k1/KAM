#!/usr/bin/env python3
"""Freeze fresh author-evidence QASPER questions with agreed yes/no labels."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq

from audit_kam_natural_data import digest, stable
from prepare_kam_qasper_evidence import norm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prior", "fixture", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    ref = json.loads(a.prior.read_text())
    source = digest(a.source)
    assert ref["source_sha256"] == source
    ids = {r["id"] for r in ref["rows"]}
    old_text = set()
    for r in ref["rows"]:
        name = r["context"][0][0].rsplit(" — ", 1)[0]
        prefix = f"In the paper {name}, "
        assert r["question"].startswith(prefix)
        old_text.add(norm(r["question"][len(prefix):]))
    held, articles = [], {}
    for paper in pq.read_table(a.source).to_pylist():
        sections = ["Abstract", *paper["full_text"]["section_name"]]
        paragraphs = [[paper["abstract"]], *paper["full_text"]["paragraphs"]]
        content = [(head or "Body", t) for head, items in zip(sections, paragraphs, strict=True)
                   for t in items if norm(t)]
        index = defaultdict(list)
        for j, (_, text) in enumerate(content):
            index[norm(text)].append(j)
        qas = paper["qas"]
        for question, qid, answer in zip(qas["question"], qas["question_id"], qas["answers"], strict=True):
            if qid in ids or norm(question) in old_text:
                continue
            valid = [note for note in answer["answer"] if not note["unanswerable"]]
            votes = [note["yes_no"] for note in valid]
            if len(votes) < 2 or None in votes or len(set(votes)) != 1:
                continue
            gold, any_evidence = set(), set()
            for note in valid:
                positions = [index[norm(t)] for t in note["evidence"]]
                any_evidence.update(j for values in positions for j in values)
                if positions and all(len(xs) == 1 for xs in positions):
                    gold.add(tuple(sorted({xs[0] for xs in positions})))
            if not gold or len(content) < 4:
                continue
            oracle = min(gold, key=lambda row: (len(row), row))
            if len(oracle) > 2:
                continue
            others = sorted((i for i in range(len(content)) if i not in any_evidence),
                            key=lambda i: stable(f"qasper-binary-sham:{qid}:{i}"))
            if len(others) < len(oracle):
                continue
            articles[paper["id"]] = content
            held.append({"id": qid, "paper": paper["id"], "question": question, "answer": bool(votes[0]),
                         "votes": len(votes), "annotations": len(answer["answer"]), "gold_sets": sorted(gold),
                         "gold": oracle, "sham": others[:len(oracle)]})
    seen = set()
    rows = []
    for q in sorted(held, key=lambda r: stable("kam-qasper-binary-v1:" + r["id"])):
        n = norm(q["question"])
        if n not in seen:
            rows.append(q)
            seen.add(n)
    counts = Counter("yes" if r["answer"] else "no" for r in rows)
    assert min(counts.values()) >= 12
    meta = {"source_sha256": source, "previous_sha256": digest(a.prior), "counts": counts}
    payload = meta | {"rows": rows, "articles": {q["paper"]: articles[q["paper"]] for q in rows}}
    a.fixture.write_bytes(gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0))
    public = meta | {"fixture_sha256": digest(a.fixture), "rows": [{k: v for k, v in r.items() if k != "question"} for r in rows]}
    a.manifest.write_bytes(gzip.compress(json.dumps(public, separators=(",", ":")).encode(), mtime=0))
    print(counts, len(payload["articles"]))


if __name__ == "__main__":
    main()
