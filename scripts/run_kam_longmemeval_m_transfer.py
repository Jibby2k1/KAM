#!/usr/bin/env python3
"""Transfer a Wikipedia-trained decomposer to sealed conversation histories."""

import argparse
from functools import lru_cache
import gzip
import json
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_longmemeval_m import by_id, chunks, iter_bank, session_ranks, turn_order
from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT
from run_kam_natural_feasibility import Encoder
from train_kam_musique_decomposer import MODEL_REVISION, REFERENCE, interfaces, predict

KEEP = 5
SOURCE_FIXTURE = "0d6a4cb740d63047f0ba03b4a06f7c1dc7563852bb37cc50e9138fd3493b72dc"


def read_json(path):
    bits = path.read_bytes()
    return json.loads(gzip.decompress(bits) if path.suffix == ".gz" else bits)


def write(path, value):
    path.write_bytes(gzip.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


def generate(a, fixture):
    model = torch.load(a.weights, weights_only=True, map_location="cpu")
    assert model["fixture_sha256"] == SOURCE_FIXTURE and model["revision"] == MODEL_REVISION
    tokenizer, decoder = interfaces(a.assets)
    decoder.load_state_dict(model["weights"])
    ids = [r["id"] for r in fixture["rows"]]
    inferred = predict(decoder, tokenizer, [r["question"] for r in fixture["rows"]])
    output = {"fixture_sha256": digest(a.fixture), "decomposer_seed": model["seed"], "weights_sha256": digest(a.weights),
              "train_fixture_sha256": SOURCE_FIXTURE, "device": torch.cuda.get_device_name(),
              "rows": [{"id": id_, "steps": s, "valid": ok} for id_, (s, ok) in zip(ids, inferred, strict=True)]}
    write(a.output, output)


def quota(n):
    return [4 // n + (j < 4 % n) for j in range(n)]


def shortlist(vector, q, groups, c):
    similarities = vector @ q
    scores = torch.full((c,), -torch.inf, device="cuda").scatter_reduce_(0, groups, similarities, reduce="amax")
    return similarities, scores


def select(indices, start, source_ids, count):
    existing = {source_ids[j] for j in start}
    add = []
    for i in indices:
        if source_ids[i] not in existing:
            existing.add(source_ids[i])
            add.append(i)
            if len(add) == count:
                break
    assert len(add) == count
    return add


def route(paragraphs, query, encoder, reader, vector, groups, positions, raw, text, mode):
    sid = query["source_session_ids"]
    output_indices = raw[:1].copy()
    bindings = []
    for step, (prompt, budget) in enumerate(zip(paragraphs, quota(len(paragraphs)), strict=True)):
        if mode == "unresolved":
            q = REFERENCE.sub("", prompt).strip()
        else:
            q = REFERENCE.sub(lambda m: bindings[int(m.group(1)) - 1], prompt).strip()
        q = q or query["question"]
        sim, scores = shortlist(vector, encoder([q], "query")[0], groups, len(sid))
        rank = session_ranks(scores, positions, sid)
        output_indices.extend(select(rank, output_indices, sid, budget))
        if mode == "bound" and step + 1 < len(paragraphs):
            best = int(sim.argmax())
            chosen_slot = int(groups[best])
            begin = int(torch.searchsorted(groups, chosen_slot))
            passage = text(query["slots"][chosen_slot])[best - begin]
            request = f"Answer the question with a short answer. Question: {q} Evidence: {passage}"
            [answer] = reader.answers([request])
            bindings.append(" ".join(answer.split()) or "unknown")
    assert len(output_indices) == len({sid[j] for j in output_indices}) == KEEP
    return output_indices


@torch.inference_mode()
def score(a, fixture):
    from run_kam_hotpot_pool_scaling import Reader

    start = perf_counter()
    fx = digest(a.fixture)
    full = fixture["rows"]
    cases = full[a.shard::a.shards]
    base = read_json(a.reference)
    index = read_json(a.bank_index)
    versions = read_json(a.assets / "revisions.json")
    assert (base["shard"], base["shards"]) == (a.shard, a.shards) and base["models"] == versions
    assert base["fixture_sha256"] == index["fixture_sha256"] == fx
    assert base["bank_sha256"] == index["bank_sha256"] == fixture["session_bank_sha256"]
    gold_free = [read_json(p) for p in a.plans]
    assert all(p["fixture_sha256"] == fx and p["train_fixture_sha256"] == SOURCE_FIXTURE for p in gold_free)
    assert all([r["id"] for r in p["rows"]] == [q["id"] for q in full] for p in gold_free)
    path_by_seed = [p["rows"][a.shard::a.shards] for p in gold_free]
    lookup = by_id(a.vectors, index, fixture, versions)
    recorded = {}
    for key, path in enumerate(a.banks):
        assert digest(path) == index["shards"][key]
        for item in iter_bank(path):
            i = item["id"]
            assert i % len(a.banks) == key and i not in recorded
            recorded[i] = item["turns"]
    assert set(recorded) == set(lookup)

    @lru_cache(maxsize=8192)
    def context(i):
        return [entry[-1] for entry in chunks(recorded[i])]

    enc = Encoder(a.assets / "encoder", 128)
    reader_path = a.assets / f"reader_base_{BASE_REVISION}"
    assert digest(reader_path / "model.safetensors") == BASE_WEIGHT
    reader = Reader(reader_path)
    base_query = enc([r["question"] for r in cases], "query")
    output = []
    for q, item, first, *predictions in zip(base_query, cases, base["rows"], *path_by_seed, strict=True):
        id_ = item["id"]
        assert id_ == first["id"] and all(r["id"] == id_ for r in predictions)
        data = [lookup[i] for i in item["slots"]]
        lengths = torch.tensor([len(d[0]) for d in data], device="cuda")
        vector = torch.cat([d[0] for d in data], 0).cuda()
        turn_ids = torch.cat([d[3] for d in data], 0).cuda().long()
        groups = torch.arange(len(data), device="cuda").repeat_interleave(lengths)
        locations = list(range(len(data)))
        norm, ranked = shortlist(vector, q, groups, len(data))
        baseline = session_ranks(ranked, locations, item["source_session_ids"])
        assert baseline == first["full"]["dense"]
        raw = {"sessions": baseline[:KEEP], "turns": turn_order(norm, groups, turn_ids, baseline[:KEEP])}
        alternatives = []
        for proposed in predictions:
            outcome = {"valid": proposed["valid"], "predicted_steps": len(proposed["steps"])}
            for mode in ("unresolved", "bound"):
                if not proposed["valid"]:
                    outcome[mode] = raw
                    continue
                ids = route(proposed["steps"], item, enc, reader, vector, groups, locations, baseline, context, mode)
                outcome[mode] = {"sessions": ids, "turns": turn_order(norm, groups, turn_ids, ids)}
            alternatives.append(outcome)
        output.append({"id": id_, "dense": raw, "decomposers": alternatives})
    write(a.output, {"fixture_sha256": fx, "reference_sha256": digest(a.reference),
                     "plans_sha256": list(map(digest, a.plans)), "seeds": [r["decomposer_seed"] for r in gold_free],
                     "models": versions | {"google/flan-t5-base": BASE_REVISION}, "shard": a.shard, "shards": a.shards,
                     "device": torch.cuda.get_device_name(), "elapsed_seconds": perf_counter() - start, "rows": output})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("generate", "score"))
    for n in ("assets", "fixture", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("weights", "reference", "bank-index"):
        p.add_argument("--" + n, type=Path)
    p.add_argument("--plans", nargs=3, type=Path)
    p.add_argument("--vectors", nargs=8, type=Path)
    p.add_argument("--banks", nargs=8, type=Path)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    fixture = read_json(a.fixture)
    (generate if a.mode == "generate" else score)(a, fixture)


if __name__ == "__main__":
    main()
