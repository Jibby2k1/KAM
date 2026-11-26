#!/usr/bin/env python3
"""Compare label-free sentence selection on frozen KAM evidence pairs."""

import argparse
from contextlib import closing
import gzip
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_gold_reader_oracle import lexical_key
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import Encoder, batches, measure_answer
from run_kam_sentence_verifier_fullwiki import SENTENCE_BREAK

ARMS = ("prefix", "lead_two", "dense_two", "bridge_two")
MAX_SENTENCES = 64


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def contexts(encoder, batch):
    questions = [case["question"] for case, _ in batch]
    documents = [document for _, pair in batch for document in pair]
    sentences = [d[1][:MAX_SENTENCES] for d in documents]
    assert all(sentences)
    flattened = [title + ". " + text for (title, _), group in zip(documents, sentences, strict=True)
                 for text in group]
    passages = encoder(flattened, "passage")
    groups = passages.split([len(group) for group in sentences])
    plain = encoder(questions, "query").repeat_interleave(2, dim=0)
    bridge = encoder([f"{question} Other evidence document: {docs[1 - i][0]}" for question, (_, docs) in
                      zip(questions, batch, strict=True) for i in (0, 1)], "query")
    selections = {}
    for arm, query in (("dense_two", plain), ("bridge_two", bridge)):
        selections[arm] = [sorted((group @ state).argsort(descending=True, stable=True)[:2].tolist())
                           for group, state in zip(groups, query, strict=True)]
    for i, (case, docs) in enumerate(batch):
        result = {"prefix": docs, "lead_two": [(title, bits[:2]) for title, bits in docs]}
        for name, selected in selections.items():
            result[name] = [(title, [bits[k] for k in selected[2 * i + j]])
                            for j, (title, bits) in enumerate(docs)]
        yield {name: {"question": case["question"], "context": item} for name, item in result.items()}


def visible_count(reader, selected, ids, evidence):
    by_id = {a["row_id"]: a for a in evidence}
    visible = 0
    for doc, (title, bits) in zip(ids, selected, strict=True):
        if doc not in by_id:
            continue
        gold = by_id[doc]
        token_ids = reader.tokenizer(title + ". " + " ".join(bits), add_special_tokens=False,
                                     max_length=192, truncation=True).input_ids
        text = lexical_key(reader.tokenizer.decode(token_ids))
        visible += sum(lexical_key(gold["source_passage"][i]) in text for i in gold["support_positions"])
    return visible


@torch.inference_mode()
def run(a):
    t0 = perf_counter()
    fixture, evidence = (json.loads(p.read_bytes()) for p in (a.fixture, a.evidence))
    replay = unpack(a.reference)
    revisions = json.loads((a.assets / "revisions.json").read_bytes())
    index_hash = digest(a.inputs / "lexical" / "completed.json")
    assert evidence["fixture_sha256"] == digest(a.fixture) and evidence["dataset_sha256"] == fixture["external_test_sha256"]
    assert all(s["fixture_sha256"] == digest(a.fixture) and s["index_sha256"] == index_hash and s["models"] == revisions
               for s in replay)
    items = fixture["rows"]["test"]
    chosen = {r["id"]: r["arms"]["conditioned_union"] for s in replay for r in s["rows"]}
    labels = {r["id"]: r["articles"] for r in evidence["rows"]}
    assert {r["id"] for r in items} == chosen.keys() == labels.keys() and len(items) == 1024
    encoder = Encoder(a.assets / "encoder", 128)
    reader = Reader(a.assets / "reader")
    prompts, rows = {}, []
    path = a.inputs / "lexical" / "articles.sqlite"
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?immutable=1", uri=True)) as db:
        db.execute("PRAGMA mmap_size=1073741824")
        for chunk in batches(items[a.shard::a.shards], 32):
            batch = []
            for case in chunk:
                pair = chosen[case["id"]]["pair"]
                docs = {i: (title, SENTENCE_BREAK.split(text) if text else [""])
                        for i, title, text in db.execute("SELECT id,title,body FROM docs WHERE id IN (?,?)", pair)}
                assert len(docs) == 2
                batch.append((case, [docs[doc] for doc in pair]))
            for case, arms in zip(chunk, contexts(encoder, batch), strict=True):
                item = chosen[case["id"]]
                support = labels[case["id"]]
                row = {"id": case["id"], "pair": item["pair"], "exact": {},
                       "valid_annotations": sum(len(a["support_positions"]) for a in support), "visible": {}}
                for arm, evidence_context in arms.items():
                    prompt = reader.prompt(evidence_context, (0, 1))
                    prompts.setdefault(prompt, []).append((row["exact"], arm, case["answer"]))
                    row["visible"][arm] = visible_count(reader, evidence_context["context"], row["pair"], support)
                rows.append(row)
    for predicted, requests in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
        for outcome, arm, expected in requests:
            outcome[arm] = bool(measure_answer(predicted, expected)["em"])
    return {"index_sha256": index_hash, "models": revisions, "fixture_sha256": digest(a.fixture),
            "reference_sha256": digest(a.reference), "evidence_sha256": digest(a.evidence),
            "shard": a.shard, "shards": a.shards, "seconds": perf_counter() - t0,
            "gpu": torch.cuda.get_device_name(), "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "assets", "fixture", "evidence", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    data = json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode()
    a.output.write_bytes(gzip.compress(data, mtime=0))


if __name__ == "__main__":
    main()
