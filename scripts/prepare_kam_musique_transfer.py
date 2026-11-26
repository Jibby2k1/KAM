#!/usr/bin/env python3
"""Freeze a supplied-pool two-hop MuSiQue-Ans transfer cohort."""

import argparse
import json
from pathlib import Path

from audit_kam_natural_data import digest, stable, words
from prepare_kam_natural_router import title_key

SOURCE_URL = "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view"
SOURCE_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"


def load_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def freeze(source, hotpot, training_partition, count, exclude=None):
    assert digest(source) == SOURCE_SHA256
    partition = json.loads(training_partition.read_text())
    assert digest(hotpot) == partition["source_sha256"]
    train_ids = set(partition["training"])
    training = [row for row in json.loads(hotpot.read_text()) if row["_id"] in train_ids]
    assert {row["_id"] for row in training} == train_ids
    training_titles = {title_key(t) for row in training for t, _ in row["context"]}
    seen = {tuple(words(row["question"])) for row in training}
    reserved = set(training_titles)
    rows = load_jsonl(source)
    excluded = set()
    if exclude:
        prior = json.loads(exclude.read_text())
        assert prior["source_sha256"] == SOURCE_SHA256
        assert prior["training_partition_sha256"] == digest(training_partition)
        excluded = {item["id"] for item in prior["rows"]}
        for row in rows:
            if row["id"] in excluded:
                reserved.update(title_key(p["title"]) for p in row["paragraphs"] if p["is_supporting"])
                seen.add(tuple(words(row["question"])))
    choices = []
    namespace = "kam-musique-gate-v1:" if exclude else "kam-musique-transfer-v1:"
    for row in sorted(rows, key=lambda row: stable(namespace + row["id"])):
        if row["id"] in excluded or not row["answerable"] or len(row["question_decomposition"]) != 2:
            continue
        paragraphs = row["paragraphs"]
        if len(paragraphs) != 20:
            continue
        assert [candidate["idx"] for candidate in paragraphs] == list(range(20))
        gold = [p for p in paragraphs if p["is_supporting"]]
        titles = {title_key(p["title"]) for p in gold}
        question = tuple(words(row["question"]))
        if len(gold) != 2 or len(titles) != 2 or titles & reserved or question in seen:
            continue
        reserved.update(titles)
        seen.add(question)
        choices.append({"id": row["id"], "gold_indices": [p["idx"] for p in gold]})
        if len(choices) == count:
            break
    if len(choices) != count:
        raise ValueError(f"Requested {count} disjoint questions, found {len(choices)}")
    fixture = {"source_url": SOURCE_URL, "source_archive_member": "data/musique_ans_v1.0_dev.jsonl",
               "source_sha256": SOURCE_SHA256, "source_count": len(rows),
               "hotpot_file_sha256": partition["source_sha256"],
               "training_partition_sha256": digest(training_partition),
               "hotpot_training_questions": len(training), "hotpot_training_candidate_titles": len(training_titles),
               "rows": choices}
    if exclude:
        fixture["excluded_fixture_sha256"] = digest(exclude)
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "hotpot", "training-partition", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--exclude-fixture", type=Path)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    selection = freeze(args.source, args.hotpot, args.training_partition, args.count, args.exclude_fixture)
    args.output.write_text(json.dumps(selection, indent=2) + "\n")


if __name__ == "__main__":
    main()
