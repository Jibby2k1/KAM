#!/usr/bin/env python3
"""Compare frozen HotpotQA routing policies on supplied MuSiQue passages."""

import argparse
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from prepare_kam_musique_transfer import load_jsonl
from run_kam_hotpot_head_to_head import adapter, route
from run_kam_hotpot_pair_scoring import PairScorer, links
from run_kam_hotpot_pool_scaling import Reader, fit_pair, pair_selection
from run_kam_natural_feasibility import Encoder, candidate_texts
from run_kam_natural_router import QueryAdapter, score

ARMS = ("single_shot", "sequential", "pair", "pair_masked")


def restore_checkpoint(weights_path, config, cfg, training_cache, title_cache, *, weights_sha256=None, device="cuda"):
    if weights_sha256 is not None and digest(weights_path) != weights_sha256:
        raise ValueError("baseline checkpoint SHA-256 mismatch")
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert weights["config_sha256"] == digest(config)
    if training_cache is not None and title_cache is not None:
        assert weights["training_cache_sha256"] == digest(training_cache)
        assert weights["title_cache_sha256"] == digest(title_cache)
    elif weights_sha256 is None or training_cache is not None or title_cache is not None:
        raise ValueError("provide both source caches or a pinned checkpoint SHA-256")
    width = weights["adapter"]["input.weight"].shape[-1]
    model = QueryAdapter(width, cfg["bottleneck"], False).to(device).eval()
    model.load_state_dict(weights["adapter"])
    scorer = PairScorer().to(device).eval()
    scorer.load_state_dict(weights["pair"])
    return model, scorer


@torch.inference_mode()
def evaluate(rows, cases, encoder, reader, model, scorer, *, link_gate=False):
    documents = [doc for case in cases for doc in candidate_texts(case)]
    keys = encoder(documents, "passage").reshape(len(cases), 20, -1)
    host_graph = links(cases)
    graph = host_graph.cuda()
    masked = torch.zeros_like(graph)
    pairs = torch.triu_indices(20, 20, 1, device="cuda")

    def request(i, arm):
        case = cases[i]
        bank = keys[i:i + 1]
        torch.cuda.synchronize()
        start = perf_counter()
        query = encoder([case["question"]], "query")
        fallback = None
        if arm == "sequential":
            selected, _ = route(encoder, model, [case], query, bank, include_frozen=False)
            pair = selected["learned_sequential"][0]
        elif arm == "single_shot":
            pair = score(model(query), bank).argsort(descending=True, stable=True)[0, :2].tolist()
        else:
            mentions = masked if arm == "pair_masked" else graph
            pair = pair_selection(model(query), bank, mentions[i:i + 1], scorer, pairs)[0].tolist()
            if arm == "link_gate":
                left, right = pair
                fallback = not bool(host_graph[i, left, right] | host_graph[i, right, left])
                if fallback:
                    selected, _ = route(encoder, model, [case], query, bank, include_frozen=False)
                    pair = selected["learned_sequential"][0]
        after_route = perf_counter()
        prediction = reader.answers([reader.prompt(case, pair)])[0]
        torch.cuda.synchronize()
        row = {"arm": arm, "pair": pair, "prediction": prediction,
               "route_seconds": after_route - start, "total_seconds": perf_counter() - start}
        if fallback is not None:
            row["sequential_fallback"] = fallback
        return row

    arms = (*ARMS, "link_gate") if link_gate else ARMS
    for arm in arms:
        request(0, arm)

    rng = Random(162095)
    output = []
    for i, row in enumerate(rows):
        records = [request(i, arm) for arm in rng.sample(arms, k=len(arms))]
        gold = row["gold_indices"]
        query = encoder([cases[i]["question"]], "query")
        affinity = score(model(query), keys[i:i + 1])[0].tolist()
        oracle = sorted(gold, key=lambda idx: (-affinity[idx], idx))
        prediction = reader.answers([reader.prompt(cases[i], oracle)])[0]
        left, right = gold
        output.append({"id": row["id"], "gold_has_title_link": bool(graph[i, left, right] | graph[i, right, left]),
                       "oracle_pair": oracle, "oracle_prediction": prediction, "arms": records})
    return output


def run(args):
    started = perf_counter()
    fixture = json.loads(args.fixture.read_text())
    assert digest(args.source) == fixture["source_sha256"]
    source = {row["id"]: row for row in load_jsonl(args.source)}
    assert len(source) == fixture["source_count"]
    cases = []
    for row in fixture["rows"]:
        sample = source[row["id"]]
        context = [(p["title"], [p["paragraph_text"]]) for p in sample["paragraphs"]]
        assert len(context) == 20 and row["gold_indices"] == [
            p["idx"] for p in sample["paragraphs"] if p["is_supporting"]]
        cases.append({"question": sample["question"], "context": context})
    cfg = json.loads(args.config.read_text())
    args.shard = 0
    cache = torch.load(args.training_cache, map_location="cpu", weights_only=True)
    titles = torch.load(args.title_cache, map_location="cpu", weights_only=True)
    title_pools = json.loads(args.title_pools.read_text())
    assert titles["pools_sha256"] == digest(args.title_pools)
    assert cache["partition_sha256"] == title_pools["training_partition_sha256"] == fixture["training_partition_sha256"]
    models = json.loads((args.assets / "revisions.json").read_text())
    assert cache["asset_revisions"] == models
    encoder, reader = Encoder(args.assets / "encoder", 64), Reader(args.assets / "reader")
    weights_path = args.weights_in or args.weights_out
    if args.weights_in:
        model, scorer = restore_checkpoint(weights_path, args.config, cfg, args.training_cache, args.title_cache)
    else:
        model = adapter(args, cfg)
        scorer = fit_pair(args, cfg, model, titles["train_mentions"].cuda(), True)
        torch.save({"config_sha256": digest(args.config), "training_cache_sha256": digest(args.training_cache),
                    "title_cache_sha256": digest(args.title_cache), "adapter": model.cpu().state_dict(),
                    "pair": scorer.cpu().state_dict()}, weights_path)
        model.cuda()
        scorer.cuda()
    output = evaluate(fixture["rows"], cases, encoder, reader, model, scorer, link_gate=args.link_gate)
    artifacts = {"source": args.source, "fixture": args.fixture, "config": args.config,
                 "training_cache": args.training_cache, "title_cache": args.title_cache,
                 "title_pools": args.title_pools, "weights": weights_path}
    result = {"status": "completed", "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "seed": cfg["seeds"][0], "elapsed_seconds": perf_counter() - started,
              "input_sha256": {name: digest(path) for name, path in artifacts.items()},
              "models": models, "rows": output}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fixture", "config", "training-cache", "title-cache", "title-pools", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--weights-in", type=Path)
    group.add_argument("--weights-out", type=Path)
    parser.add_argument("--link-gate", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
