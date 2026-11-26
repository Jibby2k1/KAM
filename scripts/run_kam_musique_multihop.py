#!/usr/bin/env python3
"""Extrapolate frozen pair scores to two-, three-, and four-document evidence sets."""

import argparse
from itertools import combinations
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_fullwiki_support_links import ContextScorer, relation_features
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader, pair_selection, prepared
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts, measure_answer
from run_kam_natural_router import score


def connected(n, edges):
    reachable = {0}
    for _ in range(n - 1):
        for i, j in edges:
            if i in reachable or j in reachable:
                reachable.update((i, j))
    return len(reachable) == n


def set_candidates(depth, k):
    subsets = torch.tensor(list(combinations(range(depth), k)), device="cuda")
    edge_index = list(combinations(range(k), 2))
    left = subsets[:, [a for a, _ in edge_index]]
    right = subsets[:, [b for _, b in edge_index]]
    trees = []
    for tree in combinations(range(len(edge_index)), k - 1):
        if connected(k, [edge_index[e] for e in tree]):
            trees.append([i in tree for i in range(len(edge_index))])
    assert len(trees) == k ** (k - 2)
    return subsets, left, right, torch.tensor(trees, device="cuda", dtype=torch.float32)


def ordered_choice(scores, subsets, affinity):
    index = int(scores.argmax())
    slots = subsets[index]
    return slots[affinity[slots].argsort(descending=True, stable=True)].tolist()


def sequential(encoder, query, adapter, case, keys, k):
    texts = candidate_texts(case)
    original = score(adapter(query), keys)
    choices = [int(original.argmax())]
    for _ in range(1, k):
        conditioned = encoder([f"{case['question']} First retrieved document: {texts[choices[-1]]}"], "query")
        total = (original + score(adapter(conditioned), keys)) * 0.5
        total[:, choices] = -torch.inf
        choices.append(int(total.argmax()))
    return choices


def reader_prompt(reader, question, context, chosen, budget):
    excerpt = []
    for i in chosen:
        title, text = context[i]
        tokens = reader.tokenizer(title + ". " + " ".join(text), add_special_tokens=False,
                                  max_length=budget, truncation=True).input_ids
        excerpt.append(reader.tokenizer.decode(tokens))
    return f"Answer the question with a short answer. Question: {question} Evidence: {' '.join(excerpt)}"


def timing(method):
    torch.cuda.synchronize()
    start = perf_counter()
    item = method()
    torch.cuda.synchronize()
    return item, perf_counter() - start


@torch.inference_mode()
def evaluate(a, rows):
    enc, reader = Encoder(a.assets / "encoder", 64), Reader(a.assets / "reader")
    cfg = read(a.config)
    adapter, original = restore_checkpoint(a.baseline, a.config, cfg, None, None,
                                           weights_sha256=a.baseline_sha256)
    frozen = torch.load(a.heads, weights_only=True, map_location="cpu")
    context_head = ContextScorer().cuda().eval()
    context_head.load_state_dict(frozen["heads"]["context_only"])
    assert frozen["provenance"]["models"] == read(a.assets / "revisions.json")
    lengths = [len(r["context"]) for r in rows]
    keys_encoded = enc([text for r in rows for text in candidate_texts(r)], "passage").split(lengths)
    queries = enc([r["question"] for r in rows], "query")
    templates = {}
    requests, results = [], []
    for i, (case, keys) in enumerate(zip(rows, keys_encoded, strict=True)):
        row_ids = case["gold"]
        k, depth = case["hops"], len(keys)
        if (depth, k) not in templates:
            cpu = torch.triu_indices(depth, depth, 1)
            templates[depth, k] = cpu, cpu.cuda(), set_candidates(depth, k)
        pairs, gpu_pairs, (subsets, ea, eb, trees) = templates[depth, k]
        keys = keys.unsqueeze(0)
        query = queries[i:i+1]
        adapted = adapter(query)

        def prep():
            graph, lexical = relation_features(case, pairs)
            mentions = graph.cuda().unsqueeze(0)
            base, features, affinity = prepared(adapted, keys, mentions, gpu_pairs)
            return graph, mentions, base[0], features[0], lexical.cuda(), affinity[0]

        (graph, mentions, pair_base, small, lexical, affinity), prep_seconds = timing(prep)
        seq, sequential_s = timing(lambda: sequential(enc, query, adapter, case, keys, k))

        def compose(head, x, tree=False):
            residual = head(pair_base, x) - pair_base
            square = torch.zeros((depth, depth), device="cuda")
            square[gpu_pairs[0], gpu_pairs[1]] = residual
            residuals = square[ea, eb]
            if tree:
                value = (residuals @ trees.T).amax(-1) * k / (2 * (k - 1))
            else:
                value = residuals.sum(-1) / (k - 1)
            return ordered_choice(20 * affinity[subsets].sum(-1) + value, subsets, affinity)

        augmented = torch.cat((small, lexical), -1)
        static = affinity.argsort(descending=True, stable=True)[:k].tolist()
        plain, plain_s = timing(lambda: compose(original, small))
        clique, clique_s = timing(lambda: compose(context_head, augmented))
        tree_set, tree_s = timing(lambda: compose(context_head, augmented, tree=True))
        if k == 2:
            explicit = pair_selection(adapted, keys, mentions, original, gpu_pairs)[0].tolist()
            assert explicit == plain or set(explicit) == set(plain)
            assert set(clique) == set(tree_set)
        gold_edges = [(m, n) for m, n in combinations(range(k), 2)
                      if graph[row_ids[m], row_ids[n]] or graph[row_ids[n], row_ids[m]]]
        arms = {}
        selections = {"single_shot_topk": static, "sequential": seq, "original_clique": plain,
                      "context_clique": clique, "context_tree": tree_set,
                      "gold": sorted(row_ids, key=lambda j: (-float(affinity[j]), j))}
        timing_by_arm = {"sequential": sequential_s, "original_clique": plain_s + prep_seconds,
                         "context_clique": clique_s + prep_seconds, "context_tree": tree_s + prep_seconds}
        for name, slots in selections.items():
            hit = set(slots) & set(row_ids)
            entry = {"selected": slots, "all": len(hit) == k, "support_count": len(hit),
                     "route_seconds": timing_by_arm.get(name)}
            arms[name] = entry
            prompt = reader_prompt(reader, case["question"], case["context"], slots, 384 // k)
            requests.append((prompt, entry, [case["answer"], *case.get("aliases", [])]))
        results.append({"id": case["id"], "hops": k, "gold": row_ids, "n_candidates": depth,
                        "n_subsets": len(subsets), "n_gold_title_links": len(gold_edges),
                        "gold_connected": connected(k, gold_edges), "arms": arms})
    prompts = list(dict.fromkeys(prompt for prompt, _, _ in requests))
    decoded = dict(zip(prompts, reader.answers(prompts), strict=True))
    for prompt, item, truths in requests:
        metrics = [measure_answer(decoded[prompt], truth) for truth in truths]
        item.update(em=max(m["em"] for m in metrics), f1=max(m["f1"] for m in metrics))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "fixture", "config", "baseline", "heads", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    data = read(a.fixture)
    rows = evaluate(a, data["rows"][a.shard::a.shards])
    save(a.output, {"fixture_sha256": digest(a.fixture), "source_sha256": data["source_sha256"],
                    "heads_sha256": digest(a.heads), "device": torch.cuda.get_device_name(),
                    "elapsed_seconds": perf_counter() - start, "rows": rows})


if __name__ == "__main__":
    main()
