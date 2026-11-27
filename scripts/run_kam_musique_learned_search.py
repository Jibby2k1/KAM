#!/usr/bin/env python3
"""Search native MuSiQue passages with learned templates and predicted bindings."""

import argparse
from functools import lru_cache
import gzip
import json
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT, binding_queries
from run_kam_musique_recursive import BUDGETS, ranked_additions
from run_kam_native_musique import DEPTH, INITIAL, corpus_and_fixture
from run_kam_natural_feasibility import Encoder


@torch.inference_mode()
def search(a, paragraphs, fixture):
    started = perf_counter()
    fixture_hash = digest(a.fixture)
    corpus_hash = fixture["corpus_sha256"]
    revisions = json.loads((a.assets / "revisions.json").read_text())
    learned = json.loads(gzip.decompress(a.generated.read_bytes()))
    assert learned["fixture_sha256"] == fixture_hash and learned["shards"] == 1
    assert learned["revision"] == revisions["google/flan-t5-small"]
    assert [q["id"] for q in learned["rows"]] == [q["id"] for q in fixture["rows"]]
    predicted = {q["id"]: q for q in learned["rows"]}
    archived = json.loads(gzip.decompress(a.baseline.read_bytes()))
    assert (archived["fixture_sha256"], archived["corpus_sha256"], archived["models"]) == (fixture_hash, corpus_hash, revisions)
    assert (archived["shard"], archived["shards"]) == (a.shard, a.shards)
    rows = archived["rows"]
    assert [q["id"] for q in rows] == [q["id"] for q in fixture["rows"][a.shard::a.shards]]
    base = a.assets / f"reader_base_{BASE_REVISION}"
    assert digest(base / "model.safetensors") == BASE_WEIGHT
    vectors = sorted((torch.load(p, weights_only=True, map_location="cpu") for p in a.vectors), key=lambda x: x["begin"])
    assert vectors[0]["begin"] == 0 and vectors[-1]["end"] == len(paragraphs)
    assert all(l["end"] == r["begin"] for l, r in zip(vectors, vectors[1:]))
    assert all(p["corpus_sha256"] == corpus_hash and p["models"] == revisions for p in vectors)
    bank = torch.cat([p["vectors"] for p in vectors], 0).cuda()
    reader = Reader(base)
    encoder = Encoder(a.assets / "encoder", 128)

    @lru_cache(maxsize=8192)
    def passage(i):
        p = paragraphs[i]
        tokens = reader.tokenizer(p["title"] + ". " + p["body"], add_special_tokens=False, max_length=128, truncation=True)
        return reader.tokenizer.decode(tokens.input_ids)

    output = {}
    groups = {n: [] for n in BUDGETS}
    for row in rows:
        id_ = row["id"]
        plans = predicted[id_]["steps"]
        if not predicted[id_]["valid"]:
            output[id_] = {"id": id_, "predicted_steps": len(plans), "fallback": True,
                           "pool": row["hybrid"], "read_from": [], "unknown_bindings": 0}
        else:
            groups[len(plans)].append((id_, plans, row["dense"][:INITIAL]))
    for nhops, data in groups.items():
        for start in range(0, len(data), 64):
            block = data[start:start+64]
            qids, templates, pools = zip(*block, strict=True)
            histories = [[] for _ in block]
            used = [[] for _ in block]
            for step, budget in enumerate(BUDGETS[nhops]):
                queries = binding_queries(templates, histories, step)
                encoded = encoder(queries, "query")
                ranks = (encoded @ bank.T).argsort(dim=-1, descending=True, stable=True)[:, :DEPTH].tolist()
                for pool, subrank in zip(pools, ranks, strict=True):
                    pool.extend(ranked_additions(subrank, pool, budget))
                if step + 1 < nhops:
                    docs = [rank[0] for rank in ranks]
                    prompts = [f"Answer the question with a short answer. Question: {q} Evidence: {passage(i)}"
                               for q, i in zip(queries, docs, strict=True)]
                    replies = reader.answers(prompts)
                    for history, answer, i, provenance in zip(histories, replies, docs, used, strict=True):
                        history.append(" ".join(answer.split()) or "unknown")
                        provenance.append(i)
            for id_, pool, chosen, values in zip(qids, pools, used, histories, strict=True):
                assert len(pool) == len(set(pool)) == DEPTH
                output[id_] = {"id": id_, "predicted_steps": nhops, "fallback": False,
                              "pool": pool, "read_from": chosen, "unknown_bindings": values.count("unknown")}
    return {"fixture_sha256": fixture_hash, "corpus_sha256": corpus_hash, "generation_sha256": digest(a.generated),
            "weights_sha256": learned["weights_sha256"], "baseline_sha256": digest(a.baseline),
            "models": revisions | {"google/flan-t5-base": BASE_REVISION}, "device": torch.cuda.get_device_name(),
            "shard": a.shard, "shards": a.shards, "elapsed_seconds": perf_counter() - started,
            "rows": [output[r["id"]] for r in rows]} | ({"oracle_depth": True} if learned.get("oracle_depth") else {})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "corpus", "fixture", "generated", "baseline", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--vectors", nargs=4, required=True, type=Path)
    p.add_argument("--shard", required=True, type=int)
    p.add_argument("--shards", required=True, type=int)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    paragraphs, fixture = corpus_and_fixture(a)
    a.output.write_bytes(gzip.compress(json.dumps(search(a, paragraphs, fixture), separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
