#!/usr/bin/env python3
"""Measure annotated IIRC span visibility at a fixed per-article text budget."""

import argparse
from collections import Counter
import json
from math import log
from pathlib import Path
import re

from audit_kam_iirc import contexts, digest, title_key

WORD = re.compile(r"\S+")
TERM = re.compile(r"\w+")
STOP = set("a an and are as at be by did do does for from had has have how in is it of on or that the their to was were what when where which who with would".split())
POLICIES = ("lead", "bm25_window", "oracle_window")
WIDTH = 160
STRIDE = 80
SOURCE_WIDTH = 100


def align(doc, annotation):
    text = annotation["text"]
    start, end = annotation["indices"]
    if doc is None or not text.strip():
        return "unresolved", None
    if 0 <= start < end <= len(doc) and doc[start:end] == text:
        status = "offset"
    else:
        start = doc.find(text)
        if start < 0 or doc.find(text, start + 1) >= 0:
            return "unresolved", None
        status, end = "unique_exact_quote", start + len(text)
    return status, (start + len(text) - len(text.lstrip()), end - len(text) + len(text.rstrip()))


def windows(doc, width, stride):
    tokens = list(WORD.finditer(doc))
    final = max(0, len(tokens) - width)
    starts = list(range(0, final + 1, stride))
    if starts[-1] != final:
        starts.append(final)
    if not tokens:
        return [(0, 0)]
    return [(tokens[i].start(), tokens[min(i + width, len(tokens)) - 1].end()) for i in starts]


def choose_bm25(doc, bounds, question):
    query = {t for t in TERM.findall(question.lower()) if t not in STOP}
    bins = [Counter(TERM.findall(doc[i:j].lower())) for i, j in bounds]
    lengths = [sum(c.values()) for c in bins]
    avg = sum(lengths) / len(bins) or 1.0
    df = Counter(t for c in bins for t in query.intersection(c))
    idf = {t: log(1 + (len(bins) - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    scores = []
    for c, length in zip(bins, lengths, strict=True):
        norm = 1.2 * (0.25 + 0.75 * length / avg)
        scores.append(sum(idf[t] * c[t] * 2.2 / (c[t] + norm) for t in sorted(query.intersection(c))))
    return max(range(len(scores)), key=scores.__getitem__)


def visibility(annotations, doc, question=None):
    aligned = [align(doc, ann) for ann in annotations]
    counts = dict(Counter(s for s, _ in aligned))
    intervals = [coords for _, coords in aligned if coords is not None]
    matched = {}
    choices = {}
    if question is None:
        first = windows(doc, SOURCE_WIDTH, SOURCE_WIDTH)[0]
        matched["source"] = sum(first[0] <= a and b <= first[1] for a, b in intervals)
    else:
        limits = windows(doc, WIDTH, STRIDE)
        totals = [sum(start <= a and b <= end for a, b in intervals) for start, end in limits]
        choices = {"lead": 0, "bm25_window": choose_bm25(doc, limits, question),
                   "oracle_window": max(range(len(totals)), key=totals.__getitem__)}
        matched = {policy: totals[pick] for policy, pick in choices.items()}
    return {"spans": len(annotations), "alignment": counts, "visible": matched, "chosen_window": choices}


def run(dev, archive, manifest):
    expected = json.loads(manifest.read_text())
    hashes = {"dev": digest(dev), "articles": digest(archive)}
    assert hashes == expected["source_sha256"]
    ids = {r["id"]: r["source_id"] for r in expected["rows"]}
    assert len(ids) == 153
    passages = json.loads(dev.read_text())
    rows, targets = [], set()
    for p in passages:
        for q in p["questions"]:
            if q["qid"] not in ids:
                continue
            assert ids[q["qid"]] == p["pid"] and q["answer"]["type"] != "none"
            notes = {}
            linked = {}
            for ann in q["context"]:
                title = ann["passage"]
                key = title_key(title)
                notes.setdefault(key, []).append(ann)
                if title != "main":
                    linked.setdefault(key, title)
                    targets.add(title)
            assert len(linked) == 2 and len(notes.get("main", []))
            rows.append((p, q, notes, linked))
    assert len(rows) == len(ids)
    keys = targets | {t.lower() for t in targets}
    articles = contexts(archive, keys, keys)
    output = []
    for p, q, notes, linked in sorted(rows, key=lambda row: row[1]["qid"]):
        linked_scores = []
        for key, title in sorted(linked.items()):
            context = articles.get(title, articles.get(title.lower()))
            assert context is not None
            linked_scores.append(visibility(notes[key], context, q["question"]))
        output.append({"id": q["qid"], "source_id": p["pid"], "answer_type": q["answer"]["type"],
                       "source": visibility(notes["main"], p["text"]), "linked": linked_scores})
    return {"source_sha256": hashes, "manifest_sha256": digest(manifest), "source_words": SOURCE_WIDTH,
            "window_words": WIDTH, "stride_words": STRIDE, "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dev", "articles", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(run(args.dev, args.articles, args.manifest), indent=2) + "\n")


if __name__ == "__main__":
    main()
