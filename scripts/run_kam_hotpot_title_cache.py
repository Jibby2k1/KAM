#!/usr/bin/env python3
"""Exact sparse title-link indexing and matched request-path timing."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from audit_kam_natural_data import digest, words
from run_kam_hotpot_head_to_head import adapter, route
from run_kam_hotpot_pair_scoring import links
from run_kam_hotpot_pool_scaling import (
    Reader, evaluate_reader, fit_pair, load_cases, pair_selection, time_one,
)
from run_kam_natural_feasibility import Encoder

ARMS = ("learned_sequential", "pair_online", "pair_cached")


def sparse_index(source, references):
    refs = sorted({tuple(ref) for pool in references for ref in pool})
    documents = {ref: source[ref[0]]["context"][ref[1]] for ref in refs}
    titles = {" ".join(words(title)) for title, _ in documents.values()}
    trie = {}
    for title in sorted(titles):
        if len(title) >= 3:
            node = trie
            for token in title.split():
                node = node.setdefault(token, {})
            node[None] = title
    index = {}
    for ref, (_, sentences) in documents.items():
        body = words(" ".join(sentences))[:120]
        mentioned = set()
        for start in range(len(body)):
            node = trie
            for position in range(start, len(body)):
                node = node.get(body[position])
                if node is None:
                    break
                if None in node:
                    mentioned.add(node[None])
        index[ref] = sorted(mentioned)
    return index


def lookup_links(case, refs, index):
    title_slots = defaultdict(list)
    for i, (title, _) in enumerate(case["context"]):
        title_slots[" ".join(words(title))].append(i)
    size = len(refs)
    row, col = [], []
    for src, key in enumerate(refs):
        for target in index[tuple(key)]:
            for dst in title_slots.get(target, ()):
                row.append(src)
                col.append(dst)
    output = torch.zeros((size, size), dtype=torch.bool)
    if row:
        output[row, col] = True
    return output


def encode(args, manifest, source, cases, references):
    start = perf_counter()
    training = torch.load(args.training_cache, map_location="cpu", weights_only=True)
    assert training["partition_sha256"] == manifest["training_partition_sha256"]
    names = {tuple(ref): source[ref[0]]["context"][ref[1]]
             for pool in references for ref in pool}
    enc = Encoder(args.assets / "encoder", 64, args.device)
    q = enc([c["question"] for c in cases], "query").cpu()
    texts = [title + ". " + " ".join(sentences) for title, sentences in names.values()]
    unique_embeddings = enc(texts, "passage").cpu()
    key_lookup = {ref: index for index, ref in enumerate(names)}
    offsets = torch.tensor([[key_lookup[tuple(ref)] for ref in pool] for pool in references])
    build_start = perf_counter()
    index = sparse_index(source, references)
    index_seconds = perf_counter() - build_start
    payload = [[list(ref), targets] for ref, targets in sorted(index.items())]
    index_bytes = len(json.dumps(payload, separators=(",", ":")).encode())
    torch.save({"queries": q, "keys": unique_embeddings[offsets],
                "train_mentions": links([source[key] for key in training["training"]["ids"]]),
                "sparse_index": payload, "pools_sha256": digest(args.pools)}, args.feature_cache)
    print(json.dumps({"encode_setup_seconds": perf_counter() - start,
                      "online_sparse_index_build_seconds": index_seconds,
                      "sparse_index_json_bytes": index_bytes, "reference_documents": len(index),
                      "directed_title_links": sum(map(len, index.values()))}), flush=True)


@torch.inference_mode()
def time_pair(enc, reader, model, scorer, case, refs, index, keys, pairs, cached):
    if keys.is_cuda:
        torch.cuda.synchronize()
    start = perf_counter()
    query = model(enc([case["question"]], "query"))
    if keys.is_cuda:
        torch.cuda.synchronize()
    feature_start = perf_counter()
    mention = (lookup_links(case, refs, index).unsqueeze(0) if cached else links([case])).to(keys.device)
    if keys.is_cuda:
        torch.cuda.synchronize()
    feature_seconds = perf_counter() - feature_start
    selected = pair_selection(query, keys, mention, scorer, pairs)[0].tolist()
    retrieval = perf_counter() - start
    reader.answers([reader.prompt(case, selected)])
    if keys.is_cuda:
        torch.cuda.synchronize()
    return {"feature_seconds": feature_seconds, "retrieval_seconds": retrieval,
            "total_seconds": perf_counter() - start, "selected": selected}


def run(args, spec, manifest, cases, references):
    start = perf_counter()
    cached = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    assert cached["pools_sha256"] == digest(args.pools)
    index = {tuple(ref): mentions for ref, mentions in cached["sparse_index"]}
    model = adapter(args, spec, args.device)
    scorer = fit_pair(args, spec, model, cached["train_mentions"].to(args.device), True)
    enc, reader = Encoder(args.assets / "encoder", 64, args.device), Reader(args.assets / "reader", args.device)
    keys, q = cached["keys"].to(args.device), cached["queries"].to(args.device)
    with torch.inference_mode():
        adapted = model(q)
    if keys.is_cuda:
        torch.cuda.synchronize()
    setup_seconds = perf_counter() - start
    records, latency, parity = [], [], {}
    rng = Random(spec["latency_seed"])
    positions = rng.sample(range(1, len(cases)), spec["latency_questions"])
    for size in manifest["sizes"]:
        small = [{**case, "context": case["context"][:size]} for case in cases]
        refs = [row[:size] for row in references]
        online_bits = links(small)
        indexed_bits = torch.stack([lookup_links(c, r, index) for c, r in zip(small, refs)])
        assert torch.equal(online_bits, indexed_bits)
        per_query_keys = keys[:, :size].contiguous()
        pair_indices = torch.triu_indices(size, size, 1, device=args.device)
        routes, _ = route(enc, model, small, q, per_query_keys, include_frozen=False)
        selections = {"learned_sequential": routes["learned_sequential"]}
        for arm, bits in (("pair_online", online_bits), ("pair_cached", indexed_bits)):
            picks = []
            for i in range(0, len(cases), 16):
                sl = slice(i, i + 16)
                picks.extend(pair_selection(adapted[sl], per_query_keys[sl], bits[sl].to(keys.device),
                                           scorer, pair_indices).tolist())
            selections[arm] = picks
        assert selections["pair_online"] == selections["pair_cached"]
        parity[str(size)] = {"mention_elements": online_bits.numel(), "ordered_selections": len(cases)}
        records.extend(evaluate_reader(reader, small, selections, size))

        def measure(position, arm):
            example = small[position]
            bank = per_query_keys[position:position + 1]
            if arm == "learned_sequential":
                return time_one(enc, reader, model, None, example, bank, pair_indices, arm)
            return time_pair(enc, reader, model, scorer, example, refs[position], index,
                             bank, pair_indices, arm == "pair_cached")

        for arm in ARMS:
            measure(0, arm)
        for pos in positions:
            for arm in rng.sample(ARMS, k=len(ARMS)):
                item = measure(pos, arm)
                latency.append({"id": small[pos]["_id"], "pool_size": size, "arm": arm, **item})
    args.output.write_text(json.dumps({
        "status": "completed", "device": torch.cuda.get_device_name() if keys.is_cuda else "CPU", "torch": torch.__version__,
        "models": json.loads((args.assets / "revisions.json").read_text()),
        "seed": spec["seeds"][args.shard], "config_sha256": digest(args.config),
        "pools_sha256": digest(args.pools), "setup_seconds": setup_seconds,
        "elapsed_seconds": perf_counter() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated() if keys.is_cuda else None,
        "link_and_route_parity": parity, "records": records, "latency": latency,
    }, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("encode", "run"))
    for name in ("source", "assets", "pools", "training-cache", "feature-cache", "config"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--shard", type=int)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    manifest = json.loads(a.pools.read_text())
    assert manifest["source_sha256"] == digest(a.source)
    spec = json.loads(a.config.read_text())
    source, cases = load_cases(a.source, manifest)
    references = [r["original"] + r["added"] for r in manifest["rows"]]
    if a.mode == "encode":
        encode(a, manifest, source, cases, references)
    else:
        run(a, spec, manifest, cases, references)


if __name__ == "__main__":
    main()
