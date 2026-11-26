#!/usr/bin/env python3
"""Train the same pair scorer on random or dense near-neighbor distractors."""

import argparse
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from prepare_kam_natural_router import title_key
from run_kam_fullwiki_native_compare import FullWikiIndex, local_title_links
from run_kam_hotpot_head_to_head import adapter
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader, fit_pair, pair_selection
from run_kam_natural_feasibility import Encoder, measure_answer


def dense_pool(index, case, query, size, blocked):
    ranks = index.search("dense", case["question"], query, size + 16)
    filtered = [row for row in ranks if row not in blocked]
    if len(filtered) < size:
        ranks = index.search("dense", case["question"], query, size + len(blocked))
        filtered = [row for row in ranks if row not in blocked]
    assert len(filtered) >= size
    return filtered[:size]


def run(a):
    started = perf_counter()
    cfg, data = read(a.config), read(a.fixture)
    manifest = read(a.inputs / "manifest.json")
    assert data["source_sha256"] == manifest["source_sha256"]
    assert data["previous_queries_sha256"] == manifest["queries_sha256"]
    training, test = data["rows"]["training"], data["rows"]["test"]
    training = [row for row in training if None not in row["gold_ids"]]
    fit_gold = {i for row in training for i in row["gold_ids"]}
    test_gold = {i for row in test for i in row["gold_ids"] if i is not None}
    assert not fit_gold & test_gold and len(fit_gold) == 2 * len(training)
    depth, test_depth = data["training_depth"], data["test_depth"]
    index = FullWikiIndex(a.inputs, manifest, cfg)
    encoder, reader = Encoder(a.assets / "encoder", 64), Reader(a.assets / "reader")
    model = adapter(a, cfg)
    original_mentions = torch.load(a.title_cache, map_location="cpu", weights_only=True)["train_mentions"].cuda()
    original = fit_pair(a, cfg, model, original_mentions, True)
    valid = index.mask.tolist()
    random = Random(331581)

    training_pools = {"random": [], "dense_hard": []}
    with torch.inference_mode():
        queries = encoder([r["question"] for r in training], "query")
        for position, case in enumerate(training):
            positive = case["gold_ids"]
            blocked = test_gold | set(positive)
            hard = dense_pool(index, case, queries[position:position+1], depth - 2, blocked)
            sampled = {}
            while len(sampled) < depth - 2:
                row = random.randrange(len(valid))
                if valid[row] and row not in blocked:
                    sampled[row] = None
            order = random.sample(range(depth), depth)
            for arm, negatives in (("random", list(sampled)), ("dense_hard", hard)):
                full = positive + negatives
                training_pools[arm].append([full[i] for i in order])

    scorers = {"original_pair": original}
    with TemporaryDirectory(prefix="kam-hard-negative-") as temporary:
        cached = Path(temporary) / "training.pt"
        for arm, pools in training_pools.items():
            gold, features = [], []
            for case, pool in zip(training, pools):
                assert len(set(pool)) == depth and not set(pool) & test_gold
                positive = set(case["gold_ids"])
                example = index.candidates(case, pool)
                assert {title_key(title) for i, (title, _) in zip(pool, example["context"])
                        if i in positive} == set(case["gold_titles"])
                features.append(local_title_links(example))
                gold.append([i in positive for i in pool])
            mentions = torch.stack(features).cuda()
            keys = index.bank[torch.tensor(pools, device="cuda")].cpu()
            torch.save({"training": {"queries": queries.cpu(), "keys": keys, "gold": torch.tensor(gold)}}, cached)
            fit_args = argparse.Namespace(**vars(a))
            fit_args.training_cache = cached
            scorers[arm] = fit_pair(fit_args, cfg, model, mentions, True)

    requests, pools = [], []
    pairs = torch.triu_indices(test_depth, test_depth, 1, device="cuda")
    with torch.inference_mode():
        for q in test:
            query = encoder([q["question"]], "query")
            ranked = dense_pool(index, q, query, test_depth, fit_gold)
            example = index.candidates(q, ranked)
            keys = index.bank[torch.tensor(ranked, device="cuda")].unsqueeze(0)
            mentions = local_title_links(example).cuda().unsqueeze(0)
            titles = [title_key(t) for t, _ in example["context"]]
            assert set(q["gold_ids"]).intersection(ranked) == {
                i for i, title in zip(ranked, titles) if title in q["gold_titles"]}
            pools.append({"id": q["id"], "row_ids": ranked, "titles": titles})
            adapted = model(query)
            for arm, scorer in scorers.items():
                pair = pair_selection(adapted, keys, mentions, scorer, pairs)[0].tolist()
                requests.append((reader.prompt(example, pair), {"id": q["id"], "arm": arm, "pair": pair}))
    prompts = list(dict.fromkeys(prompt for prompt, _ in requests))
    decoded = dict(zip(prompts, reader.answers(prompts), strict=True))
    truth = {q["id"]: q["answer"] for q in test}
    outputs = [{**row, "prediction": decoded[prompt], **measure_answer(decoded[prompt], truth[row["id"]])}
               for prompt, row in requests]
    save(a.output, {"status": "completed", "device": torch.cuda.get_device_name(), "seed": cfg["seeds"][a.shard],
                    "fixture_sha256": digest(a.fixture), "config_sha256": digest(a.config),
                    "elapsed_seconds": perf_counter() - started, "fit_ids": [r["id"] for r in training],
                    "fit_pools": training_pools, "test_pools": pools, "predictions": outputs})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "fixture", "output", "config", "training-cache", "title-cache"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
