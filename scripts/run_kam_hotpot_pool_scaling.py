#!/usr/bin/env python3
"""Evaluate pool scaling and title-link ablation with a matched reader."""

import argparse
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch.nn.functional import cross_entropy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from audit_kam_natural_data import digest
from run_kam_hotpot_head_to_head import adapter, route
from run_kam_hotpot_pair_scoring import PairScorer, links
from run_kam_natural_feasibility import Encoder, batches, candidate_texts, measure_answer
from run_kam_natural_router import score, to_device

ARMS = ("learned_sequential", "pair_full", "pair_no_titles")


def load_cases(source, manifest):
    by_id = {case["_id"]: case for case in json.loads(source.read_text())}
    expanded = []
    for row in manifest["rows"]:
        base = by_id[row["id"]]
        expected = [(base["_id"], i) for i in range(10)]
        assert [tuple(p) for p in row["original"]] == expected
        context = [by_id[id_]["context"][index] for id_, index in row["original"] + row["added"]]
        expanded.append({**base, "context": context})
    assert [c["_id"] for c in expanded] == manifest["ids"]
    return by_id, expanded


def encode(args, manifest, source, cases):
    start = perf_counter()
    enc = Encoder(args.assets / "encoder", 64)
    cached = torch.load(args.training_cache, map_location="cpu", weights_only=True)
    assert cached["partition_sha256"] == manifest["training_partition_sha256"]
    queries = enc([c["question"] for c in cases], "query").cpu()
    size = manifest["sizes"][-1]
    keys = enc([t for c in cases for t in candidate_texts(c)], "passage").reshape(len(cases), size, -1).cpu()
    encoded = {"queries": queries, "keys": keys,
               "test_mentions": links(cases), "partition_sha256": digest(args.pools),
               "train_mentions": links([source[i] for i in cached["training"]["ids"]])}
    torch.save(encoded, args.feature_cache)
    print(json.dumps({"encode_and_index_seconds": perf_counter() - start,
                      "cache_bytes": args.feature_cache.stat().st_size,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated()}), flush=True)


def prepared(query, keys, mentions, index):
    left, right = index
    affinities = score(query, keys)
    first, second = affinities[:, left], affinities[:, right]
    if mentions is None:
        zeros = torch.zeros_like(first)
        linked_either, linked_both, direction = zeros, zeros, zeros
    else:
        forward, back = mentions[:, left, right], mentions[:, right, left]
        linked_either, linked_both = (forward | back).float(), (forward & back).float()
        direction = forward * (first - second) + back * (second - first)
    cosine = (keys @ keys.transpose(-1, -2))[:, left, right]
    features = torch.stack((first + second, (first - second).abs(), cosine,
                            linked_either, linked_both, direction), dim=-1)
    return (first + second) * 20, features, affinities


def fit_pair(args, spec, model, mentions, enabled):
    device = mentions.device
    training = torch.load(args.training_cache, map_location="cpu", weights_only=True)["training"]
    q, k, gold = to_device(training, device)
    index = torch.triu_indices(k.shape[1], k.shape[1], 1, device=device)
    with torch.no_grad():
        base, x, _ = prepared(model(q), k, mentions if enabled else None, index)
        positives = gold[:, index[0]] & gold[:, index[1]]
        assert positives.sum(-1).eq(1).all()
        label = positives.float().argmax(-1)
    seed = spec["seeds"][args.shard]
    torch.manual_seed(seed + 100_000)
    pair = PairScorer().to(device)
    optimizer = torch.optim.AdamW(pair.parameters(), lr=spec["pair_learning_rate"], fused=q.is_cuda)
    rng = torch.Generator(device=device).manual_seed(seed + 140_000)
    for _ in range(spec["pair_epochs"]):
        for row in torch.randperm(len(q), generator=rng, device=device).split(spec["batch"]):
            loss = cross_entropy(pair(base[row], x[row]), label[row])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    assert torch.isfinite(loss)
    return pair.eval()


@torch.inference_mode()
def pair_selection(query, keys, mentions, scorer, index):
    base, x, affinity = prepared(query, keys, mentions, index)
    win = scorer(base, x).argmax(-1)
    left, right = index[:, win]
    row = torch.arange(len(query), device=query.device)
    left_first = affinity[row, left] >= affinity[row, right]
    return torch.stack((torch.where(left_first, left, right), torch.where(left_first, right, left)), -1)


class Reader:
    def __init__(self, path, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            path, local_files_only=True, use_safetensors=True,
        ).to(device).eval()
        self.device = device

    def prompt(self, case, titles):
        evidence = [
            case["context"][i][0] + ". " + " ".join(case["context"][i][1])
            for i in titles
        ]
        words = " ".join(
            self.tokenizer.decode(self.tokenizer(text, add_special_tokens=False, max_length=192,
                                                truncation=True).input_ids)
            for text in evidence
        )
        return f"Answer the question with a short answer. Question: {case['question']} Evidence: {words}"

    @torch.inference_mode()
    def answers(self, prompts):
        decoded = []
        for chunk in batches(prompts, 32):
            data = self.tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            tokens = self.model.generate(**data, max_new_tokens=20, do_sample=False)
            decoded.extend(self.tokenizer.batch_decode(tokens, skip_special_tokens=True))
        return decoded


def evaluate_reader(reader, cases, selected, size):
    rows, prompts = [], []
    for arm, choices in selected.items():
        for case, pair in zip(cases, choices):
            assert len(set(pair)) == 2 and min(pair) >= 0 and max(pair) < size
            gold = {title for title, _ in case["supporting_facts"]}
            hit = gold <= {case["context"][i][0] for i in pair}
            rows.append({"id": case["_id"], "pool_size": size, "arm": arm, "selected": pair,
                         "both_support": hit})
            prompts.append(reader.prompt(case, pair))
    unique = list(dict.fromkeys(prompts))
    predictions = dict(zip(unique, reader.answers(unique)))
    gold_answers = {case["_id"]: case["answer"] for case in cases}
    for row, prompt in zip(rows, prompts):
        row.update(measure_answer(predictions[prompt], gold_answers[row["id"]]))
    return rows


@torch.inference_mode()
def time_one(enc, reader, model, pair_models, case, keys, index, arm):
    if keys.is_cuda:
        torch.cuda.synchronize()
    start = perf_counter()
    query = model(enc([case["question"]], "query"))
    if arm == "learned_sequential":
        original = score(query, keys)
        first = int(original.argmax())
        document = case["context"][first]
        augmented = case["question"] + " First retrieved document: " + document[0] + ". " + " ".join(document[1])
        updated = model(enc([augmented], "query"))
        combined = original + score(updated, keys)
        combined[:, first] = -torch.inf
        selected = [first, int(combined.argmax())]
    else:
        # Recompute title matching from the raw supplied documents on
        # each request. The ablated route does not run title preparation.
        mentions = links([case]).to(keys.device) if arm == "pair_full" else None
        selected = pair_selection(query, keys, mentions, pair_models[arm], index)[0].tolist()
    retrieval = perf_counter() - start
    reader.answers([reader.prompt(case, selected)])
    if keys.is_cuda:
        torch.cuda.synchronize()
    return {"retrieval_seconds": retrieval, "total_seconds": perf_counter() - start, "selected": selected}


def run(args, spec, manifest, cases):
    data = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    assert data["partition_sha256"] == digest(args.pools)
    begun = perf_counter()
    model = adapter(args, spec)
    pair_models = {
        arm: fit_pair(args, spec, model, data["train_mentions"].cuda(), arm == "pair_full")
        for arm in ARMS if arm != "learned_sequential"
    }
    enc = Encoder(args.assets / "encoder", 64)
    reader = Reader(args.assets / "reader")
    torch.cuda.synchronize()
    setup = perf_counter() - begun
    query, all_keys, all_mentions = (data[key].cuda() for key in ("queries", "keys", "test_mentions"))
    with torch.inference_mode():
        adapted = model(query)
    records, latency = [], []
    rng = Random(spec["latency_seed"])
    positions = rng.sample(range(1, len(cases)), spec["latency_questions"])
    for size in manifest["sizes"]:
        trimmed = [{**case, "context": case["context"][:size]} for case in cases]
        index = torch.triu_indices(size, size, 1, device="cuda")
        k = all_keys[:, :size].contiguous()
        mentions = all_mentions[:, :size, :size].contiguous()
        selected, _ = route(enc, model, trimmed, query, k, include_frozen=False)
        for arm, pair in pair_models.items():
            selections = []
            for first in range(0, len(trimmed), 32):
                sl = slice(first, first + 32)
                mask = mentions[sl] if arm == "pair_full" else None
                selections.extend(pair_selection(adapted[sl], k[sl], mask, pair, index).tolist())
            selected[arm] = selections
        records.extend(evaluate_reader(reader, trimmed, selected, size))
        for arm in ARMS:
            time_one(enc, reader, model, pair_models, trimmed[0], k[:1], index, arm)
        for pos in positions:
            order = rng.sample(ARMS, k=len(ARMS))
            for arm in order:
                measurement = time_one(enc, reader, model, pair_models, trimmed[pos], k[pos:pos+1], index, arm)
                latency.append({"id": trimmed[pos]["_id"], "pool_size": size, "arm": arm, **measurement})
    args.output.write_text(json.dumps({
        "status": "completed", "scope": "fresh public development supplied-pool scaling",
        "seed": spec["seeds"][args.shard], "pools_sha256": digest(args.pools),
        "config_sha256": digest(args.config), "device": torch.cuda.get_device_name(),
        "torch": torch.__version__, "models": json.loads((args.assets / "revisions.json").read_text()),
        "setup_seconds": setup, "elapsed_seconds": perf_counter() - begun,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "records": records, "latency": latency,
    }, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("encode", "run"))
    for name in ("source", "assets", "pools", "training-cache", "feature-cache", "config"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    manifest = json.loads(a.pools.read_text())
    assert manifest["source_sha256"] == digest(a.source)
    spec = json.loads(a.config.read_text())
    source, cases = load_cases(a.source, manifest)
    if a.mode == "encode":
        encode(a, manifest, source, cases)
    else:
        run(a, spec, manifest, cases)


if __name__ == "__main__":
    main()
