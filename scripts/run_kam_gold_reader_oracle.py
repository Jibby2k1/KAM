#!/usr/bin/env python3
"""Measure the frozen reader with oracle articles, passages and support sentences."""

import argparse
from contextlib import closing
import gzip
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import torch

from audit_kam_natural_data import digest, words
from prepare_kam_natural_router import title_key
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import measure_answer

ARMS = ("indexed_gold_articles", "dataset_gold_passages", "annotated_gold_sentences")


def as_context(question, corpus, articles):
    passages = [(a["title"], a["source_passage"]) for a in articles]
    supporting = [(a["title"], [a["source_passage"][i] for i in a["support_positions"]]) for a in articles]
    return {name: {"question": question, "context": text}
            for name, text in zip(ARMS, (corpus, passages, supporting), strict=True)}


def lexical_key(text):
    return " " + " ".join(words(text)) + " "


def visibility(reader, corpus, articles):
    full = prefix = n = 0
    for context, article in zip(corpus, articles, strict=True):
        title, sentences = context
        text = title + ". " + " ".join(sentences)
        truncated = reader.tokenizer(text, add_special_tokens=False, max_length=192, truncation=True).input_ids
        source, visible = lexical_key(text), lexical_key(reader.tokenizer.decode(truncated))
        for i in article["support_positions"]:
            needle = lexical_key(article["source_passage"][i])
            assert needle.strip()
            present = needle in source
            full += present
            prefix += present and needle in visible
            n += 1
    assert prefix <= full <= n
    return {"annotated_valid": n, "in_article": full, "in_reader_prefix": prefix,
            "invalid_annotations": sum(len(a["invalid_support_positions"]) for a in articles)}


@torch.inference_mode()
def run(a):
    t0 = perf_counter()
    fixture, evidence = (json.loads(p.read_bytes()) for p in (a.fixture, a.evidence))
    meta = json.loads((a.inputs / "manifest.json").read_bytes())
    assert fixture["source_sha256"] == meta["source_sha256"]
    assert evidence["fixture_sha256"] == digest(a.fixture) and evidence["dataset_sha256"] == fixture["external_test_sha256"]
    items = fixture["rows"]["test"]
    rows = evidence["rows"]
    assert [r["id"] for r in items] == [r["id"] for r in rows] and len(rows) == 1024
    reader = Reader(a.assets / "reader")
    prompts = {}
    output = []
    path = a.inputs / "lexical" / "articles.sqlite"
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?immutable=1", uri=True)) as db:
        db.execute("PRAGMA mmap_size=1073741824")
        for case, selected in zip(items[a.shard::a.shards], rows[a.shard::a.shards], strict=True):
            articles = selected["articles"]
            ids = [d["row_id"] for d in articles]
            assert case["gold_ids"] == ids
            actual = {i: (title, [body]) for i, title, body in db.execute(
                "SELECT id,title,body FROM docs WHERE id IN (?,?)", ids)}
            corpus = [actual[i] for i in ids]
            assert [title_key(t) for t, _ in corpus] == case["gold_titles"]
            options = as_context(case["question"], corpus, articles)
            row = {"id": case["id"], "visibility": visibility(reader, corpus, articles), "exact": {}}
            for arm, context in options.items():
                if arm == "annotated_gold_sentences" and row["visibility"]["invalid_annotations"]:
                    row["exact"][arm] = None
                    continue
                prompt = reader.prompt(context, (0, 1))
                prompts.setdefault(prompt, []).append((row["exact"], arm, case["answer"]))
            output.append(row)

    for predicted, targets in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
        for result, arm, truth in targets:
            result[arm] = bool(measure_answer(predicted, truth)["em"])
    return {"fixture_sha256": digest(a.fixture), "evidence_sha256": digest(a.evidence),
            "index_sha256": digest(a.inputs / "lexical" / "completed.json"),
            "models": json.loads((a.assets / "revisions.json").read_bytes()),
            "shard": a.shard, "shards": a.shards, "seconds": perf_counter() - t0,
            "gpu": torch.cuda.get_device_name(), "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "evidence", "inputs", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    payload = json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode()
    a.output.write_bytes(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    main()
