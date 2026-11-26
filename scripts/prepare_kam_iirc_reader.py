#!/usr/bin/env python3
"""Prepare private IIRC gold-article reader inputs from verified span alignments."""

import argparse
import gzip
import json
from pathlib import Path

from audit_kam_iirc import contexts, digest, title_key
from audit_kam_iirc_visibility import align, windows, SOURCE_WIDTH, WIDTH, STRIDE


def segments(doc, marks):
    intervals = [align(doc, a)[1] for a in marks]
    assert None not in intervals
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return " ".join(doc[a:b] for a, b in merged)


def build(dev, archive, counts_path):
    replay = json.loads(gzip.decompress(counts_path.read_bytes()))
    hashes = {"dev": digest(dev), "articles": digest(archive)}
    assert replay["source_sha256"] == hashes
    rows = {r["id"]: r for r in replay["rows"]
            if not any(d["alignment"].get("unresolved", 0) for d in (r["source"], *r["linked"]))}
    assert len(rows) == 139
    cases = []
    bad = []
    titles = set()
    for p in json.loads(dev.read_text()):
        for q in p["questions"]:
            if q["qid"] not in rows:
                continue
            assert rows[q["qid"]]["source_id"] == p["pid"]
            answer = q["answer"]
            if answer["type"] == "span" and len(answer["answer_spans"]) != 1:
                bad.append(q["qid"])
                continue
            assert answer["type"] in ("span", "binary", "value")
            cases.append((p, q))
            titles.update(ann["passage"] for ann in q["context"] if ann["passage"] != "main")

    keys = titles | {t.lower() for t in titles}
    articles = contexts(archive, keys, keys)
    output = []
    for p, q in cases:
        a = q["answer"]
        marks, display = {}, {}
        for ann in q["context"]:
            title = ann["passage"]
            key = title_key(title)
            marks.setdefault(key, []).append(ann)
            display.setdefault(key, title)
        source_marks = marks.pop("main")
        s0, s1 = windows(p["text"], SOURCE_WIDTH, SOURCE_WIDTH)[0]
        links = []
        for i, (key, linked_marks) in enumerate(sorted(marks.items())):
            title = display[key]
            doc = articles.get(title, articles.get(title.lower()))
            limits = windows(doc, WIDTH, STRIDE)
            j = rows[q["qid"]]["linked"][i]["chosen_window"]["oracle_window"]
            l0, l1 = limits[0]
            o0, o1 = limits[j]
            links.append({"title": title, "lead": doc[l0:l1], "oracle_window": doc[o0:o1],
                          "gold_spans": segments(doc, linked_marks)})
        assert len(links) == 2
        gold = a["answer_spans"][0]["text"] if a["type"] == "span" else a["answer_value"]
        output.append({"id": q["qid"], "source_id": p["pid"], "question": q["question"], "answer": gold,
                       "answer_type": a["type"], "answer_unit": a.get("answer_unit", ""),
                       "source": {"title": p["title"], "lead": p["text"][s0:s1],
                                  "gold_spans": segments(p["text"], source_marks)}, "links": links})
    output.sort(key=lambda r: r["id"])
    return {"source_sha256": hashes, "span_counts_sha256": digest(counts_path),
            "excluded_multispan": sorted(bad), "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dev", "articles", "span-counts", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = build(a.dev, a.articles, a.span_counts)
    a.output.write_text(json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
