#!/usr/bin/env python3
"""Freeze new MuSiQue cohorts with separate two-, three-, and four-hop budgets."""

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_kam_natural_data import digest, stable, words
from prepare_kam_natural_router import title_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "hotpot", "prior-exclusions", "router-source", "router-partition"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--reserve", type=Path, action="append", required=True)
    parser.add_argument("--limit-per-hop", type=int, default=192)
    a = parser.parse_args()
    used_ids, seen_questions = set(), set()
    earlier_titles = set(a.prior_exclusions.read_text().splitlines())
    router = json.loads(a.router_partition.read_text())
    assert router["source_sha256"] == digest(a.router_source)
    training_ids = set(router["training"])
    excluded = {title_key(title)
                for row in json.loads(a.router_source.read_text()) if row["_id"] in training_ids
                for title, _ in row["context"]}
    for group in json.loads(a.hotpot.read_text())["rows"].values():
        for q in group:
            used_ids.add(q["id"])
            excluded.update(q["gold_titles"])
            seen_questions.add(" ".join(words(q["question"])))
    for p in a.reserve:
        for r in json.loads(p.read_text())["rows"]:
            used_ids.add(r["id"])
            if "context" in r:
                earlier_titles.update(title_key(r["context"][i][0]) for i in r["gold"])
    source = [json.loads(line) for line in a.source.read_text().splitlines() if line]
    # Earlier fixtures with just IDs still reserve every annotated supporting title.
    for row in source:
        if row["id"] in used_ids:
            earlier_titles.update(title_key(p["title"]) for p in row["paragraphs"] if p["is_supporting"])
            seen_questions.add(" ".join(words(row["question"])))
    counts = Counter()
    groups = {k: [] for k in (2, 3, 4)}
    ordered = sorted(source, key=lambda r: stable("kam-multihop-selection-v1:" + r["id"]))
    for hop in (4, 3, 2):
        for row in ordered:
            if not row["answerable"] or not row["id"].startswith(f"{hop}hop"):
                continue
            counts[f"raw_{hop}"] += 1
            if row["id"] in used_ids:
                continue
            context = [(p["title"], [p["paragraph_text"]]) for p in row["paragraphs"]]
            assert [p["idx"] for p in row["paragraphs"]] == list(range(len(context)))
            gold = [p["idx"] for p in row["paragraphs"] if p["is_supporting"]]
            assert 15 <= len(context) <= 20
            titles = {title_key(context[p][0]) for p in gold}
            qkey = " ".join(words(row["question"]))
            if len(gold) != hop or len(titles) < hop or titles & excluded or qkey in seen_questions:
                continue
            groups[hop].append({"id": row["id"], "question": row["question"], "answer": row["answer"],
                                "hops": hop, "context": context, "gold": gold,
                                "gold_titles": sorted(titles), "previous_title_overlap": bool(titles & earlier_titles),
                                "aliases": row.get("answer_aliases", [])})
            seen_questions.add(qkey)
            if len(groups[hop]) == a.limit_per_hop:
                break
    selected = [r for k in (2, 3, 4) for r in groups[k]]
    assert min(map(len, groups.values())) >= 48, {hop: len(r) for hop, r in groups.items()}
    output = {"source_sha256": digest(a.source), "excluded_titles_sha256": digest(a.prior_exclusions),
              "router_partition_sha256": digest(a.router_partition),
              "reserved": [digest(p) for p in a.reserve], "reviewed_counts": dict(counts),
              "counts": {hop: len(rows) for hop, rows in groups.items()}, "rows": selected}
    a.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(output["counts"])


if __name__ == "__main__":
    main()
