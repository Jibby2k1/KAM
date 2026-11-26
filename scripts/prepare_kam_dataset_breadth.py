#!/usr/bin/env python3
"""Freeze title-disjoint supplied-context and external news study fixtures."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from audit_kam_natural_data import digest, stable, words
from prepare_kam_natural_router import title_key


def eligible(rows, seen_questions, seen_titles, limit, name):
    counts = Counter(source=len(rows))
    picked = []
    # Freeze order independently of model output and link availability.
    for row in sorted(rows, key=lambda q: stable(f"kam-breadth-v1:{name}:{q['id']}")):
        context = row["context"]
        gold = row["gold"]
        if len(gold) != 2 or len(set(gold)) != 2 or not all(0 <= n < len(context) for n in gold):
            counts["invalid_two_support"] += 1
            continue
        titles = {title_key(context[i][0]) for i in gold}
        if len(titles) != 2 or titles & seen_titles:
            counts["prior_or_duplicate_title"] += 1
            continue
        question = " ".join(words(row["question"]))
        if not question or question in seen_questions:
            counts["prior_or_duplicate_question"] += 1
            continue
        if len(context) < 3:
            counts["too_few_candidates"] += 1
            continue
        picked.append(row)
        seen_questions.add(question)
        seen_titles.update(titles)
        if len(picked) == limit:
            break
    counts["selected"] = len(picked)
    return picked, dict(counts)


def supplied(a, known_titles, known_questions):
    reservations = [r for path in a.reserve for r in json.loads(path.read_text())["rows"]]
    previously_used = {r["id"] for r in reservations}
    for row in reservations:
        if "context" in row and "gold" in row:
            known_titles.update(title_key(row["context"][i][0]) for i in row["gold"])
        if "question" in row:
            known_questions.add(" ".join(words(row["question"])))
    if a.dataset == "twowiki":
        raw = pq.read_table(a.source).to_pylist()
        rows = []
        for q in raw:
            context = list(zip(q["context"]["title"], q["context"]["sentences"], strict=True))
            facts = set(q["supporting_facts"]["title"])
            gold = [i for i, (title, _) in enumerate(context) if title in facts]
            if len(facts) == 2 and q["id"] not in previously_used and set(context[i][0] for i in gold) == facts:
                rows.append({"id": q["id"], "question": q["question"], "answer": q["answer"],
                             "kind": q["type"], "context": context, "gold": gold})
    else:
        raw = [json.loads(line) for line in a.source.read_text().splitlines() if line]
        rows = []
        for q in raw:
            if not q["answerable"] or q["id"] in previously_used:
                continue
            paragraphs = q["paragraphs"]
            assert [p["idx"] for p in paragraphs] == list(range(len(paragraphs)))
            rows.append({"id": q["id"], "question": q["question"], "answer": q["answer"],
                         "aliases": q.get("answer_aliases", []), "kind": q["id"].split("__")[0],
                         "context": [(p["title"], [p["paragraph_text"]]) for p in paragraphs],
                         "gold": [p["idx"] for p in paragraphs if p["is_supporting"]]})
    selected, counts = eligible(rows, known_questions, known_titles, a.limit, a.dataset)
    return {"raw_rows": len(raw), "counts": counts, "rows": selected}


def news(a, known_questions):
    raw = json.loads(a.source.read_text())
    corpus = json.loads(a.corpus.read_text())
    assert len({row["url"] for row in corpus}) == len(corpus)
    known_urls = {row["url"] for row in corpus}
    picked, used = [], set()
    for q in sorted(raw, key=lambda row: stable("kam-breadth-news-v1:" + row["query"])):
        urls = {ev["url"] for ev in q["evidence_list"]}
        qkey = " ".join(words(q["query"]))
        if len(urls) != 2 or not urls <= known_urls or qkey in known_questions or urls & used:
            continue
        picked.append({"id": stable(q["query"])[:20], "question": q["query"], "answer": q["answer"],
                       "kind": q["question_type"], "evidence": q["evidence_list"]})
        used.update(urls)
        known_questions.add(qkey)
        if len(picked) == a.limit:
            break
    return {"raw_rows": len(raw), "counts": {"selected": len(picked)}, "rows": picked,
            "corpus_sha256": digest(a.corpus), "articles": len(corpus)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("twowiki", "musique", "news"))
    for name in ("source", "output", "prior-hotpot", "prior-exclusions"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--reserve", type=Path, action="append", default=[])
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--limit", type=int, default=768)
    a = parser.parse_args()
    hotpot = json.loads(a.prior_hotpot.read_text())["rows"]
    known_titles = set(a.prior_exclusions.read_text().splitlines())
    known_titles.update(t for rows in hotpot.values() for q in rows for t in q["gold_titles"])
    known_questions = {" ".join(words(q["question"])) for rows in hotpot.values() for q in rows}
    payload = news(a, known_questions) if a.dataset == "news" else supplied(a, known_titles, known_questions)
    assert len(payload["rows"]) >= 96, payload["counts"]
    metadata = {"dataset": a.dataset, "source_sha256": digest(a.source),
                "prior_hotpot_sha256": digest(a.prior_hotpot), "prior_exclusions_sha256": digest(a.prior_exclusions),
                "reservation_sha256": [digest(path) for path in a.reserve]}
    a.output.write_text(json.dumps({**metadata, **payload}, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
