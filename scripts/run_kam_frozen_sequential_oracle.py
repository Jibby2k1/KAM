#!/usr/bin/env python3
"""Run learned sequential retrieval and the same reader on persisted Wikipedia pools."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_hotpot_head_to_head import route
from run_kam_hotpot_independent import read
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, measure_answer
from run_kam_sentence_verifier_fullwiki import pool_text, selected_vectors


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


@torch.inference_mode()
def run(a):
    start = perf_counter()
    index_sha = digest(a.inputs / "lexical" / "completed.json")
    config, fixture, manifest = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json"))
    pools = unpack(a.pools)
    assert fixture["source_sha256"] == manifest["source_sha256"]
    assert pools["fixture_sha256"] == digest(a.fixture) and pools["index_sha256"] == index_sha
    prior = unpack(a.broad_replay)
    assert all(s["pools_sha256"] == digest(a.pools) and s["index_sha256"] == index_sha for s in prior)
    assert all(s["models"] == read(a.assets / "revisions.json") for s in prior)
    reference = {r["id"]: r for s in prior for r in s["rows"] if r["split"] == "test"}
    cases = fixture["rows"]["test"]
    assert len(cases) == len(reference) == 1024
    assert set(reference) == {q["id"] for q in cases}
    cases = cases[a.shard::a.shards]
    selected = [i for case in cases for i in pools["rows"][case["id"]]]
    vectors, by_id = selected_vectors(a.inputs, manifest, config["encoding_parts_per_row_group"], selected, "cuda")
    encoder = Encoder(a.assets / "encoder", 128)
    adapter, _ = restore_checkpoint(a.baseline, a.config, config, None, None, weights_sha256=a.baseline_sha256)
    reader = Reader(a.assets / "reader")

    examples, banks = [], []
    with sqlite3.connect(f"file:{a.inputs / 'lexical' / 'articles.sqlite'}?immutable=1", uri=True) as db:
        db.execute("PRAGMA mmap_size=1073741824")
        for q in cases:
            ranked = pools["rows"][q["id"]]
            assert len(ranked) == len(set(ranked)) == 64
            examples.append(pool_text(db, q["question"], ranked))
            banks.append([by_id[i] for i in ranked])
    banks = vectors[torch.tensor(banks, device="cuda")]
    queries = encoder([q["question"] for q in cases], "query")
    sequential, _ = route(encoder, adapter, examples, queries, banks, include_frozen=False)

    rows, prompts = [], {}
    for case, example, second in zip(cases, examples, sequential["learned_sequential"], strict=True):
        id_ = case["id"]
        prior_row = reference[id_]
        ranked = pools["rows"][id_]
        pool_sha = sha256(json.dumps(ranked, separators=(",", ":")).encode()).hexdigest()
        assert pool_sha == prior_row["pool_hash"]
        broad = prior_row["arms"]["broad"]["pair"]
        gold = set(case["gold_ids"])
        options = {}
        for name, pair in (("broad", broad), ("sequential", second)):
            assert len(pair) == len(set(pair)) == 2
            support = {ranked[j] for j in pair} == gold
            outcome = {"pair": pair, "support": support}
            options[name] = outcome
            prompts.setdefault(reader.prompt(example, pair), []).append((outcome, case["answer"]))
        assert options["broad"]["support"] == prior_row["arms"]["broad"]["support"]
        rows.append({"id": id_, "pool_hash": pool_sha, "arms": options})

    for predicted, requests in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
        for outcome, truth in requests:
            outcome["exact"] = bool(measure_answer(predicted, truth)["em"])
    return {"shard": a.shard, "shards": a.shards, "gpu": torch.cuda.get_device_name(),
            "fixture_sha256": digest(a.fixture), "pools_sha256": digest(a.pools),
            "broad_replay_sha256": digest(a.broad_replay), "baseline_sha256": a.baseline_sha256,
            "index_sha256": index_sha, "seconds": perf_counter() - start, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "config", "fixture", "pools", "broad-replay", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    payload = json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode()
    a.output.write_bytes(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    main()
