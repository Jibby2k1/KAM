#!/usr/bin/env python3
"""Freeze the IIRC two-linked-source development cohort for a routing replay."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from audit_kam_iirc import contexts, digest, title_key


def link_context(text, spans):
    excerpts = []
    for start, end in spans[:3]:
        before = " ".join(text[:start].split()[-20:])
        anchor = text[start:end]
        after = " ".join(text[end:].split()[:20])
        excerpts.append(f"{before} {anchor} {after}".strip())
    return " ".join(excerpts)


def build(questions, archive, split="dev", with_link_context=False):
    passages = json.loads(questions.read_text())
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
        links, spans = {}, {}
        for link in p["links"]:
            target = link["target"]
            text = articles.get(target, articles.get(target.lower()))
            if text is None:
                continue
            title = title_key(target)
            links.setdefault(title, (target, " ".join(text.split()[:160])))
            if with_link_context:
                spans.setdefault(title, []).append(link["indices"])
        order = sorted(links)
        for q, supports in subset:
            gold = [i for i, t in enumerate(order) if t in supports]
            assert len(gold) == 2 and len(order) > 2
            row = {"id": q["qid"], "source_id": p["pid"], "question": q["question"],
                   "source_excerpt": " ".join(p["text"].split()[:100]),
                   "candidates": [links[t] for t in order], "gold_indices": gold}
            if with_link_context:
                row["link_context"] = [link_context(p["text"], spans[t]) for t in order]
            rows.append(row)
    rows.sort(key=lambda r: r["id"])
    assert len(rows) == len({r["id"] for r in rows}) == {"dev": 153, "train": 1200}[split]
    description = "development" if split == "dev" else "training"
    return {"scope": f"IIRC official {description} questions with exactly two annotated linked sources",
            "source_sha256": {split: digest(questions), "articles": digest(archive)},
            "source_articles": len(selected), "candidate_count": dict(sorted(Counter(len(r["candidates"]) for r in rows).items())),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("articles", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    input_split = parser.add_mutually_exclusive_group(required=True)
    input_split.add_argument("--dev", type=Path)
    input_split.add_argument("--train", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--with-link-context", action="store_true")
    args = parser.parse_args()
    split = "dev" if args.dev else "train"
    fixture = build(getattr(args, split), args.articles, split, args.with_link_context)
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
