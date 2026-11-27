#!/usr/bin/env python3
"""Build a complete paragraph bank and freeze fresh MuSiQue-Ans queries."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from audit_kam_natural_data import digest, stable, words
from prepare_kam_natural_router import title_key

SIZES = {2: 224, 3: 192, 4: 160}
MEMBERS = {name: f"data/musique_ans_v1.0_{name}.jsonl" for name in ("train", "dev")}


def reserve(paths):
    ids, wordings = set(), set()
    for p in paths:
        for row in json.loads(p.read_text())["rows"]:
            ids.add(row["id"])
            wordings.add(tuple(words(row["question"])))
    return ids, wordings


def canonical_paragraph(row):
    return title_key(row["title"]), " ".join(words(row["paragraph_text"]))


def build(source, reservations, target):
    reserved_ids, reserved_words = reserve(reservations)
    prior = {"questions": len(reserved_ids), "normalized_wordings": len(reserved_words),
             "fixture_sha256": [digest(p) for p in reservations]}
    corpus, pending, input_digests = {}, defaultdict(list), {}
    source_counters = {split: Counter() for split in MEMBERS}
    seen_train = 0
    with ZipFile(source) as archive, target.open("wb") as out, gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0) as data:
        for split, member in MEMBERS.items():
            counter = source_counters[split]
            seen_train = len(corpus) if split == "dev" else seen_train
            checks = sha256()
            with archive.open(member) as raw:
                for line in raw:
                    checks.update(line)
                    if not line.strip():
                        continue
                    example = json.loads(line)
                    counter["questions"] += 1
                    gold = []
                    for p in example["paragraphs"]:
                        title, description = canonical_paragraph(p)
                        assert title and description
                        fingerprint = sha256(f"{title}\x1f{description}".encode()).digest()
                        if fingerprint not in corpus:
                            i = corpus[fingerprint] = len(corpus)
                            record = {"id": i, "title": p["title"], "body": p["paragraph_text"]}
                            data.write((json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n").encode())
                            counter["new_paragraphs"] += 1
                        counter["paragraphs"] += 1
                        if p["is_supporting"]:
                            gold.append(corpus[fingerprint])
                    if split == "dev" and example["answerable"] and example["id"] not in reserved_ids:
                        hops = int(example["id"][0])
                        if hops in SIZES and len(set(gold)) == hops:
                            pending[hops].append({"id": example["id"], "question": example["question"],
                                                  "hops": hops, "gold": sorted(gold)})
            input_digests[split] = checks.hexdigest()
    selected, chains = [], set()
    for hops, n in SIZES.items():
        count = 0
        for row in sorted(pending[hops], key=lambda r: stable("kam-native-musique-query-v1:" + r["id"])):
            wording = tuple(words(row["question"]))
            chain = tuple(row["gold"])
            if wording in reserved_words or len(wording) < 1 or chain in chains:
                continue
            row["gold_already_in_train_paragraph_bank"] = sum(i < seen_train for i in chain)
            selected.append(row)
            reserved_words.add(wording)
            chains.add(chain)
            count += 1
            if count == n:
                break
        assert count == n, (hops, count)
    selected.sort(key=lambda q: stable("kam-native-musique-order-v1:" + q["id"]))
    assert len(selected) == len({x["id"] for x in selected}) == sum(SIZES.values())
    return {"source_sha256": digest(source), "member_sha256": input_digests, "reservations": prior,
            "sources": source_counters, "documents": len(corpus), "training_documents": seen_train,
            "corpus_sha256": digest(target), "rows": selected}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "corpus", "fixture"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reserve", action="append", required=True, type=Path)
    a = p.parse_args()
    data = build(a.source, a.reserve, a.corpus)
    a.fixture.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n")
    print({k: v for k, v in data.items() if k != "rows"})


if __name__ == "__main__":
    main()
