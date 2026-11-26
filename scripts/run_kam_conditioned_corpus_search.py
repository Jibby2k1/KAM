#!/usr/bin/env python3
"""Hold the first hop fixed and search the full indexed corpus for the second document."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_hotpot_independent import read
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts, measure_answer
from run_kam_natural_router import score
from run_kam_sentence_verifier_fullwiki import pool_text, SENTENCE_BREAK

DEPTH = 64
BATCH = 16
BLOCK = 262144
CORPUS_ARMS = ("corpus_unconditioned", "corpus_conditioned")


def top_corpus(index, queries, first, depth=DEPTH):
    """Exact FP32 batched dot products with a bounded-size score workspace."""
    n = len(queries)
    first = torch.tensor(first, device=queries.device)
    values = queries.new_full((n, depth), -torch.inf)
    ids = first.new_full((n, depth), -1)
    for start in range(0, len(index.bank), BLOCK):
        stop = min(start + BLOCK, len(index.bank))
        scores = queries @ index.bank[start:stop].T
        scores.masked_fill_(~index.mask[start:stop], -torch.inf)
        local = first - start
        in_block = (local >= 0) & (local < stop - start)
        # Mask the fixed first document only when it resides in this block.
        pos = local.clamp(0, stop - start - 1).unsqueeze(1)
        prior = scores.gather(1, pos).masked_fill_(in_block.unsqueeze(1), -torch.inf)
        scores.scatter_(1, pos, prior)
        local_values, local_ids = scores.topk(min(depth, stop - start), sorted=False)
        values, order = torch.cat((values, local_values), 1).topk(depth)
        ids = torch.cat((ids, local_ids + start), 1).gather(1, order)
    assert torch.isfinite(values).all()
    return ids.tolist(), values[:, :2].tolist()


def new_context(db, row_id):
    title, body = db.execute("SELECT title,body FROM docs WHERE id=?", (row_id,)).fetchone()
    return title, SENTENCE_BREAK.split(body)


def load_gzip(path):
    return json.loads(gzip.decompress(path.read_bytes()))


@torch.inference_mode()
def run(a):
    start = perf_counter()
    cfg, fixture, manifest = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json"))
    sealed, prior = load_gzip(a.pools), load_gzip(a.reference)
    assets = read(a.assets / "revisions.json")
    fixture_sha, pools_sha = digest(a.fixture), digest(a.pools)
    index_sha = digest(a.inputs / "lexical" / "completed.json")
    assert manifest["source_sha256"] == fixture["source_sha256"]
    assert sealed["fixture_sha256"] == fixture_sha and sealed["index_sha256"] == index_sha
    assert all(s["fixture_sha256"] == fixture_sha and s["pools_sha256"] == pools_sha
               and s["index_sha256"] == index_sha and s["models"] == assets
               and s["baseline_sha256"] == a.baseline_sha256 for s in prior)
    reference = {r["id"]: r for s in prior for r in s["rows"] if r["split"] == "test"}
    cases = fixture["rows"]["test"]
    assert len(cases) == len(reference) == 1024 and set(reference) == {q["id"] for q in cases}
    cases = cases[a.shard::a.shards]
    index = FullWikiIndex(a.inputs, manifest, cfg)
    index_loaded = perf_counter()
    encoder = Encoder(a.assets / "encoder", 128)
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    reader = Reader(a.assets / "reader")

    rows, prompts = [], {}
    search_seconds = 0.
    for start_at in range(0, len(cases), BATCH):
        chunk = cases[start_at:start_at+BATCH]
        ranked = [sealed["rows"][q["id"]] for q in chunk]
        assert all(len(p) == len(set(p)) == DEPTH for p in ranked)
        examples = [pool_text(index.db, q["question"], p) for q, p in zip(chunk, ranked, strict=True)]
        query = adapter(encoder([q["question"] for q in chunk], "query"))
        bank = index.bank[torch.tensor(ranked, device="cuda")]
        first_scores = score(query, bank)
        first_slots = first_scores.argmax(1, keepdim=True)
        first_slots_list = first_slots.flatten().tolist()
        augmented = [
            f"{c['question']} First retrieved document: {candidate_texts(c)[first]}"
            for c, first in zip(examples, first_slots_list, strict=True)
        ]
        next_query = adapter(encoder(augmented, "query"))
        same_scores = (first_scores + score(next_query, bank)) * .5
        same_scores.scatter_(1, first_slots, -torch.inf)
        second_slots_list = same_scores.argmax(1).tolist()
        first_ids = [pool[i] for pool, i in zip(ranked, first_slots_list, strict=True)]
        # The blended query's dot product equals the same-pool sequential scoring rule.
        corpus_queries = torch.cat((query, (query + next_query) * .5))
        torch.cuda.synchronize()
        search_start = perf_counter()
        found, top_scores = top_corpus(index, corpus_queries, first_ids * len(CORPUS_ARMS))
        search_seconds += perf_counter() - search_start

        for j, (case, example, pool) in enumerate(zip(chunk, examples, ranked, strict=True)):
            id_ = case["id"]
            old = reference[id_]
            pool_sha = sha256(json.dumps(pool, separators=(",", ":")).encode()).hexdigest()
            assert old["pool_hash"] == pool_sha
            first = first_slots_list[j]
            second = second_slots_list[j]
            assert old["arms"]["sequential"]["pair"] == [first, second], id_
            corpus = {arm: found[j + len(chunk) * i] for i, arm in enumerate(CORPUS_ARMS)}
            assert all(len(p) == len(set(p)) == DEPTH and first_ids[j] not in p for p in corpus.values())
            arms = {}
            for arm, pair in (("broad", old["arms"]["broad"]["pair"]), ("same_pool_sequential", [first, second])):
                arms[arm] = {"pair": [pool[p] for p in pair]}
                prompts.setdefault(reader.prompt(example, pair), []).append((arms[arm], case["answer"]))
            for arm, hits in corpus.items():
                row_id = hits[0]
                first_context = example["context"][first]
                context = example["context"][pool.index(row_id)] if row_id in pool else new_context(index.db, row_id)
                selected = {"question": case["question"], "context": [first_context, context]}
                arms[arm] = {"pair": [first_ids[j], row_id]}
                prompts.setdefault(reader.prompt(selected, [0, 1]), []).append((arms[arm], case["answer"]))
            margins = {arm: round(top_scores[j + len(chunk) * i][0] - top_scores[j + len(chunk) * i][1], 8)
                       for i, arm in enumerate(CORPUS_ARMS)}
            rows.append({"id": id_, "pool_hash": pool_sha, "arms": arms, "corpus_hits": corpus, "top1_margin": margins})

    for predicted, requests in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
        for arm, truth in requests:
            arm["exact"] = bool(measure_answer(predicted, truth)["em"])
    return {"fixture_sha256": fixture_sha, "pools_sha256": pools_sha, "reference_sha256": digest(a.reference),
            "index_sha256": index_sha, "baseline_sha256": a.baseline_sha256, "config_sha256": digest(a.config),
            "models": assets, "indexed_documents": index.meta["indexed"], "candidate_depth": DEPTH,
            "shard": a.shard, "shards": a.shards, "gpu": torch.cuda.get_device_name(),
            "index_load_seconds": index_loaded - start, "batched_search_seconds": search_seconds,
            "seconds": perf_counter() - start, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "config", "fixture", "pools", "reference", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    a = parser.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    result = json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode()
    a.output.write_bytes(gzip.compress(result, mtime=0))


if __name__ == "__main__":
    main()
