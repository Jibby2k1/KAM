#!/usr/bin/env python3
"""Freeze a shared external donor bank and matched ten-passage prefixes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import pyarrow.parquet as pq

from audit_kam_natural_data import digest, stable, words


def norm(text):
    return " ".join(words(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("twowiki", "musique", "source", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--capacity", type=int, default=8192)
    a = p.parse_args()
    groups = {name: json.loads(getattr(a, name).read_text())["rows"] for name in ("twowiki", "musique")}
    excluded = {norm(title) for cases in groups.values() for row in cases for title, _ in row["context"]}
    excluded_ids = {case["id"] for cases in groups.values() for case in cases}
    raw = pq.read_table(a.source, columns=["id", "context"]).to_pylist()
    donor_by_title = {}
    for row in raw:
        if row["id"] in excluded_ids:
            continue
        docs = row["context"]
        for title, paragraphs in zip(docs["title"], docs["sentences"], strict=True):
            name = norm(title)
            text = " ".join(p.strip() for p in paragraphs)
            if name in excluded or not name or len(words(text)) < 16:
                continue
            signature = stable(row["id"] + title + text)
            candidate = (signature, title, text)
            if name not in donor_by_title or signature < donor_by_title[name][0]:
                donor_by_title[name] = candidate
    top = sorted(donor_by_title.values(), key=lambda x: stable("kam-matched-donor-v1:" + x[0]))[:a.capacity]
    assert len(top) == a.capacity
    donors = [{"id": sha256(f"{title}\n{text}".encode()).hexdigest(), "title": title, "text": text}
              for _, title, text in top]
    normalized = {}
    for name, cases in groups.items():
        normalized[name] = []
        for r in cases:
            if len(r["context"]) < 10:
                continue
            rng = Random(int(stable("kam-matched-depth-v1:" + r["id"])[:12], 16))
            indices = list(r["gold"]) + rng.sample(sorted(set(range(len(r["context"]))) - set(r["gold"])), 8)
            rng.shuffle(indices)
            assert len(indices) == len(set(indices)) == 10
            normalized[name].append({"id": r["id"], "question": r["question"], "answer": r["answer"],
                                     "kind": r["kind"], "gold": [i for i, j in enumerate(indices) if j in r["gold"]],
                                     "context": [r["context"][i] for i in indices],
                                     "original_indices": indices, "original_depth": len(r["context"])})
    fixture = {"derived_from": {name: digest(getattr(a, name)) for name in ("twowiki", "musique", "source")},
               "donors": donors, "rows": normalized}
    assert all(len(group) >= 200 for group in normalized.values())
    a.output.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n")
    print({"donors": len(donors), "datasets": {n: len(g) for n, g in normalized.items()}})


if __name__ == "__main__":
    main()
