#!/usr/bin/env python3
"""Read matched 2Wiki prompts with the two required article identities supplied."""

import argparse
from collections import defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import torch
import pyarrow.parquet as pq

from audit_kam_cross_dataset_wikipedia import tokens, wordspan
from audit_kam_natural_data import digest, stable
from prepare_kam_natural_router import title_key
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import measure_answer


def reference_rows(path, ids):
    columns = ["id", "question", "answer", "context", "supporting_facts"]
    data = pq.read_table(path, columns=columns, filters=[("id", "in", sorted(ids))]).to_pylist()
    assert {row["id"] for row in data} == ids and len(data) == len(ids)
    return {row["id"]: row for row in data}


def contexts(case, reference, database, *, raw_reference=False):
    titles = defaultdict(set)
    facts = reference["supporting_facts"]
    for title, sent in zip(facts["title"], facts["sent_id"], strict=True):
        titles[title].add(sent)
    official = dict(zip(reference["context"]["title"], reference["context"]["sentences"], strict=True))
    by_title = {u["title_sha256"]: u["corpus_id"] for u in case["units"]}
    ordering = sorted(titles, key=lambda t: stable(case["id"] + "\n" + t))
    assert len(ordering) == 2 and case["question"] == reference["question"]
    arms = {"native_leads": [], "supplied_context": [], "annotated_sentences": []}
    required = []
    for title in ordering:
        title_hash = sha256(title_key(title).encode()).hexdigest()
        doc, native = database.execute("SELECT title, body FROM docs WHERE id=?", (by_title[title_hash],)).fetchone()
        assert title_key(title) == title_key(doc)
        annotated = [official[title][j] for j in sorted(titles[title])]
        required.extend(annotated if raw_reference else map(tokens, annotated))
        arms["native_leads"].append((title, [native]))
        arms["supplied_context"].append((title, official[title]))
        arms["annotated_sentences"].append((title, annotated))
    return arms, required


@torch.inference_mode()
def run(a):
    started = perf_counter()
    fixture = json.loads(a.fixture.read_text())
    assert fixture["source_sha256"] == digest(a.source)
    cases = fixture["rows"][a.shard::a.shards]
    references = reference_rows(a.source, {r["id"] for r in cases})
    reader = Reader(a.assets / "reader")
    rows, requests = [], []
    db = (a.inputs / "lexical/articles.sqlite").resolve()
    with sqlite3.connect(f"{db.as_uri()}?immutable=1", uri=True) as database:
        for case in cases:
            source = references[case["id"]]
            arms, units = contexts(case, source, database)
            outcomes = {}
            for method, ctx in arms.items():
                prompt = reader.prompt({"question": case["question"], "context": ctx}, [0, 1])
                actual = reader.tokenizer.decode(reader.tokenizer(prompt, truncation=True, max_length=512).input_ids,
                                                 skip_special_tokens=True)
                visible = tokens(actual.split("Evidence:", 1)[1])
                outcome = {"visible_units": sum(wordspan(visible, unit)[0] for unit in units)}
                outcomes[method] = outcome
                requests.append((prompt, source["answer"], outcome))
            rows.append({"id": case["id"], "units": len(units),
                         "answer_kind": "binary" if source["answer"].strip().casefold() in ("yes", "no") else "other",
                         "arms": outcomes})
    prompts = list(dict.fromkeys(p for p, _, _ in requests))
    predicted = dict(zip(prompts, reader.answers(prompts), strict=True))
    for prompt, answer, record in requests:
        record.update(measure_answer(predicted[prompt], answer))
    return {"fixture_sha256": digest(a.fixture), "source_sha256": fixture["source_sha256"],
            "corpus": fixture["corpus"], "models": json.loads((a.assets / "revisions.json").read_text()),
            "device": torch.cuda.get_device_name(), "shard": a.shard, "shards": a.shards,
            "elapsed_seconds": perf_counter() - started, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "source", "fixture", "inputs", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    a.output.write_bytes(gzip.compress(json.dumps(run(a), separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
