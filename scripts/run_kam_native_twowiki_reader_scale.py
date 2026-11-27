#!/usr/bin/env python3
"""Change reader capacity on the same selected and gold 2Wiki prompts."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import torch
from transformers import AutoTokenizer

from audit_kam_natural_data import digest
from run_kam_hotpot_pool_scaling import Reader
from run_kam_native_twowiki_reader import contexts, reference_rows
from run_kam_natural_feasibility import measure_answer
from run_kam_sentence_verifier_fullwiki import SENTENCE_BREAK

ARMS = ("frozen", "refit", "gold_native", "gold_annotations")
BASE_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
BASE_WEIGHT = "1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8"


@torch.inference_mode()
def run(a):
    started = perf_counter()
    fixture = json.loads(a.fixture.read_text())
    picks = json.loads(gzip.decompress(a.pairs.read_bytes()))
    fixed = json.loads(gzip.decompress(a.gold.read_bytes()))
    metadata = json.loads((a.inputs / "lexical/completed.json").read_text())
    revisions = json.loads((a.assets / "revisions.json").read_text())
    assert fixture["source_sha256"] == digest(a.source) == picks["source_sha256"] == fixed["source_sha256"]
    assert fixture["corpus"] == picks["corpus"] == fixed["corpus"]
    assert all(metadata[k] == v for k, v in fixture["corpus"].items())
    assert digest(a.fixture) == picks["fixture_sha256"] == fixed["fixture_sha256"]
    assert (a.shard, a.shards) == (picks["shard"], picks["shards"]) == (fixed["shard"], fixed["shards"])
    assert revisions == picks["models"] == fixed["models"]
    small = a.assets / "reader"
    path = small if a.model == "small" else a.assets / f"reader_base_{BASE_REVISION}"
    model_revision = revisions["google/flan-t5-small"] if a.model == "small" else BASE_REVISION
    if a.model != "small":
        assert digest(path / "model.safetensors") == BASE_WEIGHT
    reader = Reader(path)
    if a.model == "base_bf16":
        reader.model.to(dtype=torch.bfloat16)
    tokenizer_small = AutoTokenizer.from_pretrained(small, local_files_only=True)
    cases = fixture["rows"][a.shard::a.shards]
    refs = reference_rows(a.source, {q["id"] for q in cases})
    bank = (a.inputs / "lexical/articles.sqlite").resolve()
    rows, requests = [], []
    with sqlite3.connect(f"{bank.as_uri()}?immutable=1", uri=True) as db:
        for q, pair, gold in zip(cases, picks["rows"], fixed["rows"], strict=True):
            id_ = q["id"]
            assert id_ == pair["id"] == gold["id"]
            source = refs[id_]
            true = {u["corpus_id"] for u in q["units"]}
            oracle, _ = contexts(q, source, db)
            inputs = {"gold_native": oracle["native_leads"], "gold_annotations": oracle["annotated_sentences"]}
            for name in ARMS[:2]:
                docs = [db.execute("SELECT title, body FROM docs WHERE id=?", (i,)).fetchone()
                        for i in pair[name]["pair"]]
                inputs[name] = [(title, SENTENCE_BREAK.split(body)) for title, body in docs]
            row = {"id": id_, "arms": {}}
            for arm in ARMS:
                ctx = {"question": q["question"], "context": inputs[arm]}
                prompt = reader.prompt(ctx, [0, 1])
                expected_from_small = (pair[arm]["em"] if arm in ARMS[:2]
                                       else gold["arms"]["native_leads" if arm == "gold_native"
                                                        else "annotated_sentences"]["em"])
                record = {"support": arm not in ARMS[:2] or set(pair[arm]["pair"]) == true,
                          "archived_small_em": expected_from_small}
                row["arms"][arm] = record
                requests.append((prompt, source["answer"], record))
            rows.append(row)

    prompts = list(dict.fromkeys(p for p, _, _ in requests))
    given = reader.tokenizer(prompts, truncation=True, max_length=512).input_ids
    matching = tokenizer_small(prompts, truncation=True, max_length=512).input_ids
    assert given == matching
    prompt_hash = sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest()
    token_hash = sha256(json.dumps(given, separators=(",", ":")).encode()).hexdigest()
    # Warm a fixed 32-prompt batch and then measure all unique prompts.
    reader.answers(prompts[:32])
    torch.cuda.synchronize()
    before_decode = perf_counter()
    predicted = dict(zip(prompts, reader.answers(prompts), strict=True))
    torch.cuda.synchronize()
    seconds = perf_counter() - before_decode
    for prompt, answer, record in requests:
        record.update(measure_answer(predicted[prompt], answer))
    return {"fixture_sha256": digest(a.fixture), "pairs_sha256": digest(a.pairs), "fixed_sha256": digest(a.gold),
            "source_sha256": fixture["source_sha256"], "corpus": fixture["corpus"],
            "model": a.model, "revision": model_revision, "parameters": sum(p.numel() for p in reader.model.parameters()),
            "decode_seconds": seconds, "total_seconds": perf_counter() - started, "prompts": len(prompts),
            "prompt_sha256": prompt_hash, "token_sha256": token_hash,
            "device": torch.cuda.get_device_name(), "shard": a.shard, "shards": a.shards, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "fixture", "source", "pairs", "gold", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--shard", required=True, type=int)
    p.add_argument("--shards", required=True, type=int)
    p.add_argument("--model", required=True, choices=("small", "base", "base_bf16"))
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    a.output.write_bytes(gzip.compress(json.dumps(run(a), separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
