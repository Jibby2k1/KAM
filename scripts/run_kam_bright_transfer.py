#!/usr/bin/env python3
"""Compare one-pass and conditioned retrieval on independent BRIGHT domains."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_bright_bm25 import corpus
from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import Encoder
from train_kam_musique_decomposer import REFERENCE, interfaces, model_source, predict

LIMIT = 64
RRF_DEPTH = 128
RRF_SCALE = 60
TRAINING_FIXTURE_SHA256 = "0d6a4cb740d63047f0ba03b4a06f7c1dc7563852bb37cc50e9138fd3493b72dc"


def order(matrix, cases):
    scores = matrix.clone()
    for i, c in enumerate(cases):
        if c["excluded"]:
            scores[i, c["excluded"]] = -torch.inf
    return scores.argsort(dim=-1, descending=True, stable=True)[:, :RRF_DEPTH].tolist()


def rrf(ranks, weights):
    scores = defaultdict(float)
    for rank, w in zip(ranks, weights, strict=True):
        for j, i in enumerate(rank, 1):
            scores[i] += w / (RRF_SCALE + j)
    return sorted(scores, key=lambda i: (-scores[i], i))[:LIMIT]


def loaded(inputs, fixture):
    rows = []
    for shard, item in enumerate(fixture["parts"]):
        p = inputs / f"corpus-{shard}.jsonl.gz"
        assert digest(p) == item["sha256"]
        block = list(corpus(p))
        assert len(block) == item["high"] - item["low"]
        assert len(rows) == item["low"]
        rows += block
    assert len(rows) == fixture["documents"]
    return rows


def encode(a, fixture, revisions):
    part = fixture["parts"][a.shard]
    path = a.inputs / f"corpus-{a.shard}.jsonl.gz"
    assert digest(path) == part["sha256"]
    docs = list(corpus(path))
    assert len(docs) == part["high"] - part["low"]
    enc = Encoder(a.assets / "encoder", 128)
    ts = perf_counter()
    texts = [row["content"] for row in docs]
    # Equal token inputs in a length-bucketed order reduce padding.
    grouping = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors = enc([texts[i] for i in grouping], "passage", max_length=192).cpu()
    output = torch.empty_like(vectors)
    output[torch.tensor(grouping)] = vectors
    torch.save({"provenance": part, "index": a.shard, "revisions": revisions, "vectors": output,
                "fixture_sha256": digest(a.inputs/"fixture.json"), "device": torch.cuda.get_device_name(),
                "index_seconds": perf_counter()-ts}, a.output)


def snippet(tokenizer, text, max_length, tail=False):
    token_ids = tokenizer(text, add_special_tokens=False, verbose=False).input_ids
    block = token_ids[-max_length:] if tail else token_ids[:max_length]
    return tokenizer.decode(block)


def learned_plan(a, weights, questions, cases, baseline, enc, bank, docs, reader, lexical):
    fit = torch.load(weights, map_location="cpu", weights_only=True)
    assert fit["fixture_sha256"] == TRAINING_FIXTURE_SHA256 and not fit["binding_tokens"]
    size = fit["model_size"]
    tokenizer, model = interfaces(a.assets, size)
    assert model_source(a.assets, size)[-1] == fit["revision"]
    model.load_state_dict(fit["weights"])
    torch.cuda.synchronize()
    start = perf_counter()
    predicted = predict(model, tokenizer, questions)
    torch.cuda.synchronize()
    plan_seconds = perf_counter() - start
    model = None
    choices = [[] for _ in cases]
    bindings = [[] for _ in cases]
    started = perf_counter()
    for step in range(4):
        active = [(i, path[step]) for i, (path, valid) in enumerate(predicted) if valid and step < len(path)]
        if not active:
            continue
        queries = [REFERENCE.sub(lambda m: bindings[i][int(m.group(1))-1], plan).strip() for i, plan in active]
        vectors = enc(queries, "query")
        chosen = order(vectors @ bank.T, [cases[i] for i, _ in active])
        needed, prompts = [], []
        for (i, _), question, rank in zip(active, queries, chosen, strict=True):
            choices[i].append(rank)
            path = predicted[i][0]
            if step + 1 < len(path):
                text = snippet(reader.tokenizer, docs[rank[0]]["content"], 128)
                prompts.append(f"Answer the question with a short answer. Question: {question} Evidence: {text}")
                needed.append(i)
        for i, answer in zip(needed, reader.answers(prompts), strict=True):
            bindings[i].append(" ".join(answer.split()) or "unknown")
    found = []
    combined = []
    for actual, old, lex in zip(choices, baseline, lexical, strict=True):
        ranks = [old, *actual]
        weights_for_fusion = [1.] + [1/len(actual) for _ in actual]
        found.append(rrf(ranks, weights_for_fusion))
        if a.hybrid_augmentation:
            combined.append(rrf([lex, old, *actual], [1., .5, *[.5 / len(actual) for _ in actual]]) if actual
                            else rrf([lex, old], [1., 1.]))
    torch.cuda.synchronize()
    return {"weights_sha256": digest(weights), "training_fixture_sha256": fit["fixture_sha256"],
            "size": size, "seed": fit["seed"], "plan_seconds": plan_seconds,
            "retrieval_and_binding_seconds": perf_counter()-started,
            "valid": [valid for _, valid in predicted], "steps": [len(paths) for paths in choices], "rank": found,
            "augmented": combined}


def search(a, fixture, revisions):
    assert a.parts and a.lexical and a.weights
    fx = digest(a.inputs/"fixture.json")
    records = sorted([torch.load(f, map_location="cpu", weights_only=True) for f in a.parts], key=lambda q: q["index"])
    for src, ref in zip(records, fixture["parts"], strict=True):
        assert src["fixture_sha256"] == fx and src["revisions"] == revisions and src["provenance"] == ref
    bank = torch.cat([r["vectors"] for r in records]).cuda()
    assert len(bank) == fixture["documents"]
    docs = loaded(a.inputs, fixture)
    lexical = json.loads(gzip.decompress(a.lexical.read_bytes()))
    assert lexical["fixture_sha256"] == fx and lexical["documents"] == fixture["documents"]
    cases = fixture["rows"]
    assert [c["id"] for c in cases] == [v["id"] for v in lexical["rows"]]
    queries = [c["question"] for c in cases]
    enc = Encoder(a.assets / "encoder", 128)
    head = enc(queries, "query")
    values = head @ bank.T
    dense = order(values, cases)
    tail = enc([snippet(enc.tokenizer, q, 175, tail=True) for q in queries], "query")
    two = order(torch.maximum(values, tail @ bank.T), cases)
    clipped = [snippet(enc.tokenizer, q, 85) for q in queries]
    anchor = [snippet(enc.tokenizer, docs[p[0]]["content"], 85) for p in dense]
    source = [f"{question} First retrieved passage: {passage}" for question, passage in zip(clipped, anchor, strict=True)]
    next_q = enc(source, "query")
    conditioned = order(.5 * (values + next_q @ bank.T), cases)
    reader_dir = a.assets / f"reader_base_{BASE_REVISION}"
    assert digest(reader_dir/"model.safetensors") == BASE_WEIGHT
    reader = Reader(reader_dir)
    bm25 = [r["ids"] for r in lexical["rows"]]
    by_model = [learned_plan(a, w, queries, cases, dense, enc, bank, docs, reader, bm25) for w in a.weights]
    assert len({(r["size"], r["seed"]) for r in by_model}) == len(by_model)
    rows = []
    for i, q in enumerate(cases):
        lex = lexical["rows"][i]["ids"]
        options = {"bm25": lex[:LIMIT], "dense": dense[i][:LIMIT], "one_pass_windows": two[i][:LIMIT],
                   "conditioned": conditioned[i][:LIMIT], "hybrid": rrf((dense[i], lex), (1., 1.))}
        if a.hybrid_augmentation:
            options |= {"hybrid_windows": rrf([lex, dense[i], two[i]], [1., .5, .5]),
                        "hybrid_anchor": rrf([lex, dense[i], conditioned[i]], [1., .5, .5])}
        for model in by_model:
            name = f"{model['size']}/{model['seed']}"
            options[name] = model["rank"][i]
            if a.hybrid_augmentation:
                options["hybrid_plan/" + name] = model["augmented"][i]
        assert all(len(v) == len(set(v)) == LIMIT and not set(v).intersection(q["excluded"]) for v in options.values())
        rows.append({"id": q["id"], "ranks": options,
                     "valid_plans": {f"{m['size']}/{m['seed']}": m["valid"][i] for m in by_model},
                     "plan_searches": {f"{m['size']}/{m['seed']}": m["steps"][i] for m in by_model}})
    output = {"fixture_sha256": fx, "task": fixture["task"], "device": torch.cuda.get_device_name(),
              "revisions": revisions | {"google/flan-t5-base": BASE_REVISION},
              "weight_records": [{k: v for k, v in m.items() if k not in ("rank", "steps", "valid", "augmented")} for m in by_model],
              "index_gpu_seconds": [q["index_seconds"] for q in records], "bm25_build_seconds": lexical["build_seconds"],
              "documents": fixture["documents"], "rows": rows, "hybrid_augmentation": a.hybrid_augmentation}
    a.output.write_bytes(gzip.compress(json.dumps(output, separators=(",", ":")).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("encode", "search"))
    for opt in ("assets", "inputs", "output"):
        p.add_argument("--" + opt, type=Path, required=True)
    p.add_argument("--lexical", type=Path)
    p.add_argument("--parts", type=Path, nargs=4)
    p.add_argument("--weights", type=Path, nargs="+")
    p.add_argument("--shard", type=int)
    p.add_argument("--hybrid-augmentation", action="store_true")
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    fixture = json.loads((a.inputs/"fixture.json").read_text())
    revisions = json.loads((a.assets/"revisions.json").read_text())
    with torch.inference_mode():
        (encode if a.mode == "encode" else search)(a, fixture, revisions)


if __name__ == "__main__":
    main()
