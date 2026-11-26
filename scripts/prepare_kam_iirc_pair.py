#!/usr/bin/env python3
"""Freeze the IIRC two-linked-source development cohort for a routing replay."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from audit_kam_iirc import contexts, digest, title_key


def build(dev, archive):
    passages = json.loads(dev.read_text())
    selected = []
    for p in passages:
        subset = []
        for q in p["questions"]:
            if q["answer"]["type"] == "none":
                continue
            supporting = {title_key(span["passage"]) for span in q["context"] if span["passage"] != "main"}
            if len(supporting) == 2:
                subset.append((q, supporting))
        if subset:
            selected.append((p, subset))

    targets = {link["target"] for p, _ in selected for link in p["links"]}
    keys = targets | {t.lower() for t in targets}
    articles = contexts(archive, keys, keys)
    rows = []
    for p, subset in selected:
        links = {}
        for link in p["links"]:
            target = link["target"]
            text = articles.get(target, articles.get(target.lower()))
            if text is None:
                continue
            title = title_key(target)
            links.setdefault(title, (target, " ".join(text.split()[:160])))
        order = sorted(links)
        for q, supports in subset:
            gold = [i for i, t in enumerate(order) if t in supports]
            assert len(gold) == 2 and len(order) > 2
            rows.append({"id": q["qid"], "source_id": p["pid"], "question": q["question"],
                         "source_excerpt": " ".join(p["text"].split()[:100]),
                         "candidates": [links[t] for t in order], "gold_indices": gold})
    rows.sort(key=lambda r: r["id"])
    assert len(rows) == len({r["id"] for r in rows}) == 153
    return {"scope": "IIRC official development questions with exactly two annotated linked sources",
            "source_sha256": {"dev": digest(dev), "articles": digest(archive)},
            "source_articles": len(selected), "candidate_count": dict(sorted(Counter(len(r["candidates"]) for r in rows).items())),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dev", "articles", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    fixture = build(args.dev, args.articles)
    payload = (json.dumps(fixture, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    args.output.write_bytes(payload)
    if args.manifest:
        rows = fixture.pop("rows")
        fixture["fixture_sha256"] = sha256(payload).hexdigest()
        fixture["rows"] = [
            {"id": r["id"], "source_id": r["source_id"],
             "pool_sha256": sha256(json.dumps(r["candidates"], ensure_ascii=False).encode()).hexdigest(),
             "candidates": len(r["candidates"]), "gold": r["gold_indices"]}
            for r in rows
        ]
        args.manifest.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
