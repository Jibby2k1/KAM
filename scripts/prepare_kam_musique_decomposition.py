#!/usr/bin/env python3
"""Freeze title-disjoint fit and validation data for a MuSiQue decomposer."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from audit_kam_natural_data import digest, stable, words
from prepare_kam_native_musique import MEMBERS, title_key
from run_kam_musique_recursive import REFERENCE
from run_kam_native_musique import corpus_and_fixture

COUNTS = {2: 2048, 3: 1152, 4: 384}
VALID = {4: 48, 3: 64, 2: 96}


def normalized(question):
    return " ".join(words(question))


def title_sha(corpus_title):
    return sha256(title_key(corpus_title).encode()).hexdigest()


def prepare(a, articles, fixture):
    assert digest(a.source) == fixture["source_sha256"]
    test = fixture["rows"]
    held_ids = {q["id"] for q in test}
    held_titles = {title_sha(articles[x]["title"]) for q in test for x in q["gold"]}
    held_questions = {normalized(q["question"]) for q in test}
    train_member = sha256()
    eligible = {k: [] for k in COUNTS}
    blocked = Counter()
    with ZipFile(a.source) as archive, archive.open(MEMBERS["train"]) as source:
        for line in source:
            train_member.update(line)
            row = json.loads(line)
            decomp = row["question_decomposition"]
            if len(decomp) not in COUNTS or not row["answerable"]:
                continue
            n = len(decomp)
            word = normalized(row["question"])
            titles = {title_sha(x["title"]) for x in row["paragraphs"] if x["is_supporting"]}
            if row["id"] in held_ids or word in held_questions or titles & held_titles:
                blocked[n] += 1
                continue
            steps = [s["question"].strip() for s in decomp]
            if any(not (p and all(1 <= int(j) <= i for j in REFERENCE.findall(p))) for i, p in enumerate(steps)):
                continue
            if not word or not titles or not any(REFERENCE.search(p) for p in steps):
                continue
            eligible[n].append({"id": row["id"], "question": row["question"], "word": word, "steps": steps,
                                "support_title_sha256": sorted(titles)})
    assert train_member.hexdigest() == fixture["member_sha256"]["train"]

    sizes = {n: len(rows) for n, rows in eligible.items()}
    four_hop_title_counts = Counter(t for q in eligible[4] for t in q["support_title_sha256"])
    selected, seen_words = {}, set(held_questions)
    validation_titles = set()
    for split, quotas in (("validation", VALID), ("fit", COUNTS)):
        forbidden = held_titles if split == "validation" else held_titles | validation_titles
        chosen = []
        for n, target in quotas.items():
            count = 0
            def priority(row):
                # Four-hop fit data is scarce. Validation uses less-shared
                # evidence so that disjointness removes fewer fit questions.
                overlap = sum(four_hop_title_counts[t] for t in row["support_title_sha256"]) if split == "validation" else 0
                return overlap, stable(f"kam-music-{split}-v1:" + row["id"])

            for q in sorted(eligible[n], key=priority):
                if q["word"] in seen_words or set(q["support_title_sha256"]) & forbidden:
                    continue
                if split == "validation":
                    validation_titles.update(q["support_title_sha256"])
                seen_words.add(q["word"])
                chosen.append(q)
                count += 1
                if count == target:
                    break
            assert count == target, (split, n, count, target, sizes)
        chosen.sort(key=lambda x: stable("kam-music-split-order-v1:" + x["id"]))
        selected[split] = [{k: v for k, v in row.items() if k != "word"} for row in chosen]
    assert {r["id"] for r in selected["fit"]}.isdisjoint(r["id"] for r in selected["validation"])
    splits = {
        k: {title for q in rows for title in q["support_title_sha256"]}
        for k, rows in selected.items()
    }
    assert splits["fit"].isdisjoint(splits["validation"])
    assert (splits["fit"] | splits["validation"]).isdisjoint(held_titles)
    metadata = {"fixture_sha256": digest(a.fixture), "source_sha256": fixture["source_sha256"],
                "training_member_sha256": train_member.hexdigest(),
                "held_out_title_count": len(held_titles), "split_title_count": {k: len(v) for k, v in splits.items()},
                "eligible_by_hop": sizes, "excluded_by_hop": dict(blocked), "quotas": {"fit": COUNTS, "validation": VALID}}
    return metadata | {"splits": selected}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "source", "fixture", "output", "manifest"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    articles, cfg = corpus_and_fixture(a)
    panel = prepare(a, articles, cfg)
    a.output.write_bytes(gzip.compress(json.dumps(panel, separators=(",", ":")).encode(), mtime=0))
    index = {name: [{**{k: r[k] for k in ("id", "support_title_sha256")}, "hops": len(r["steps"])}
                    for r in rows] for name, rows in panel.pop("splits").items()}
    a.manifest.write_bytes(gzip.compress(json.dumps(panel | {"data_sha256": digest(a.output), "ids": index},
                                                    separators=(",", ":")).encode(), mtime=0))
    print(panel)


if __name__ == "__main__":
    main()
