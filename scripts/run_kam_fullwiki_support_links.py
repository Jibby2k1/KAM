#!/usr/bin/env python3
"""Fit and test a question-conditioned link scorer on fixed full-corpus pools."""

import argparse
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from audit_kam_natural_data import digest, words
from prepare_kam_natural_router import title_key
from run_kam_fullwiki_hard_negatives import dense_pool
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_hotpot_head_to_head import route
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pair_scoring import PairScorer
from run_kam_hotpot_pool_scaling import Reader, prepared
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, measure_answer

STOP = set("a an and are as at be by did do does for from had has have how in is it of on or that the their to was were what when where which who with would".split())
FIT_ARMS = ("matched_plain", "context_only", "support_link")


def relation_features(example, pairs):
    """Use exact title mentions and question terms within 12 tokens of each mention."""
    question = set(words(example["question"])) - STOP
    scale = max(len(question), 1)
    names, bodies, trie = [], [], {}
    for i, (title, text) in enumerate(example["context"]):
        name = words(title)
        body = words(" ".join(text))[:120]
        names.append(set(name))
        bodies.append(body)
        if len(" ".join(name)) >= 3:
            node = trie
            for token in name:
                node = node.setdefault(token, {})
            node.setdefault(None, []).append(i)
    n = len(bodies)
    graph = torch.zeros((n, n), dtype=torch.bool)
    local = torch.zeros((n, n))
    coverage = [question & (set(body) | name) for body, name in zip(bodies, names)]
    for i, body in enumerate(bodies):
        for first, token in enumerate(body):
            node = trie.get(token)
            last = first
            while node is not None:
                for j in node.get(None, ()):
                    graph[i, j] = True
                    before = body[max(0, first - 12):first]
                    after = body[last + 1:last + 13]
                    hits = len((set(before + after) & question) - names[j]) / scale
                    local[i, j] = max(float(local[i, j]), hits)
                last += 1
                node = node.get(body[last]) if last < len(body) else None
    left, right = pairs
    content = torch.tensor([(len(coverage[i] | coverage[j]) / scale,
                             min(len(coverage[i]), len(coverage[j])) / scale,
                             len(coverage[i] & coverage[j]) / scale)
                            for i, j in zip(left.tolist(), right.tolist(), strict=True)])
    fwd, rev = local[left, right], local[right, left]
    return graph, torch.cat((content, torch.stack((torch.maximum(fwd, rev), fwd + rev), -1)), -1)


class ContextScorer(PairScorer):
    def __init__(self):
        super().__init__()
        self.layers[0] = nn.Linear(11, 24)


def fit_heads(base, features, label, linked, cfg):
    seeds = {"matched_plain": 49331, "context_only": 51407, "support_link": 51407}
    eligible = linked.gather(1, label[:, None]).squeeze(1) & (linked.sum(dim=1) > 1)
    heads = {}
    for arm in FIT_ARMS:
        torch.manual_seed(seeds[arm])
        head = (PairScorer() if arm == "matched_plain" else ContextScorer()).cuda()
        x = features[..., :6] if arm == "matched_plain" else features
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["pair_learning_rate"], fused=True)
        rng = torch.Generator(device="cuda").manual_seed(19113)
        for _ in range(cfg["pair_epochs"]):
            for batch in torch.randperm(len(label), generator=rng, device="cuda").split(cfg["batch"]):
                logits = head(base[batch], x[batch])
                loss = cross_entropy(logits, label[batch])
                if arm == "support_link":
                    valid = eligible[batch]
                    # Noneligible rows retain their full logits and contribute zero.
                    masked = logits.masked_fill(valid[:, None] & ~linked[batch], -torch.inf)
                    auxiliary = cross_entropy(masked, label[batch], reduction="none")
                    loss = loss + (auxiliary * valid).sum() / valid.sum().clamp_min(1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        assert torch.isfinite(loss)
        heads[arm] = head.eval()
    return heads, int(eligible.sum())


def selections(logits, pairs, affinity, mask):
    winner = int(logits.argmax())
    left, right = pairs[:, winner].tolist()
    selected = [left, right] if affinity[left] >= affinity[right] else [right, left]
    linked_winner = int(logits.masked_fill(~mask, -torch.inf).argmax()) if mask.any() else None
    return selected, linked_winner


def run(a):
    start = perf_counter()
    cfg, fixture = read(a.config), read(a.fixture)
    manifest = read(a.inputs / "manifest.json")
    assert fixture["source_sha256"] == manifest["source_sha256"]
    assert fixture["previous_queries_sha256"] == manifest["queries_sha256"] == digest(a.inputs / "queries.json")
    train, test = (fixture["rows"][group] for group in ("training", "test"))
    assert len({r["id"] for r in train + test}) == len(train) + len(test)
    assert len({t for r in train + test for t in r["gold_titles"]}) == 2 * (len(train) + len(test))
    depth = fixture["test_depth"]
    assert depth == fixture["training_depth"] == 64
    torch.set_float32_matmul_precision("highest")
    index = FullWikiIndex(a.inputs, manifest, cfg)
    encoder = Encoder(a.assets / "encoder", 64)
    model, original = restore_checkpoint(a.weights, a.config, cfg, a.training_cache, a.title_cache,
                                         weights_sha256=a.baseline_sha256)
    pairs = torch.triu_indices(depth, depth, 1)
    gpu_pairs = pairs.cuda()
    weights_digest = digest(a.weights)
    provenance = {"fixture_sha256": digest(a.fixture), "config_sha256": digest(a.config),
                  "weights_sha256": weights_digest, "device": torch.cuda.get_device_name(),
                  "torch": str(torch.__version__), "models": read(a.assets / "revisions.json"),
                  "index_sha256": digest(a.inputs / "lexical/completed.json")}
    test_gold = {i for q in test for i in q["gold_ids"] if i is not None}

    def pool(case, ranked, query):
        example = index.candidates(case, ranked)
        titles = [title_key(t) for t, _ in example["context"]]
        gold = tuple(i for i, t in enumerate(titles) if t in case["gold_titles"])
        assert {ranked[i] for i in gold} == set(ranked).intersection(case["gold_ids"])
        graph, context = relation_features(example, pairs)
        graph = graph.cuda()
        keys = index.bank[torch.tensor(ranked, device="cuda")].unsqueeze(0)
        base, x, affinity = prepared(model(query), keys, graph.unsqueeze(0), gpu_pairs)
        links = graph[gpu_pairs[0], gpu_pairs[1]] | graph[gpu_pairs[1], gpu_pairs[0]]
        return example, keys, gold, base[0], x[0], torch.cat((x[0], context.cuda()), -1), affinity[0], links

    if a.mode == "fit":
        rng = Random(42937)
        training, audit = [], []
        pair_slots = {tuple(pair): j for j, pair in enumerate(pairs.T.tolist())}
        with torch.no_grad():
            included = [q for q in train if None not in q["gold_ids"]]
            queries = encoder([q["question"] for q in included], "query")
            for i, case in enumerate(included):
                query = queries[i:i + 1]
                ranked = dense_pool(index, case, query, depth, test_gold)
                retrieved = len(set(ranked).intersection(case["gold_ids"]))
                ranked = list(dict.fromkeys(case["gold_ids"] + ranked))[:depth]
                rng.shuffle(ranked)
                assert len(ranked) == depth and not set(ranked) & test_gold
                _, _, gold, base, _, x, _, links = pool(case, ranked, query)
                label = pair_slots[gold]
                training.append((base, x, torch.tensor(label, device="cuda"), links))
                audit.append({"id": case["id"], "row_ids": ranked, "naturally_retrieved_gold": retrieved,
                              "gold_linked": bool(links[label]), "linked_pairs": int(links.sum())})
        heads, eligible = fit_heads(*(torch.stack(t) for t in zip(*training, strict=True)), cfg)
        torch.save({"provenance": provenance, "heads": {arm: h.cpu().state_dict() for arm, h in heads.items()}},
                   a.checkpoint)
        save(a.output, {**provenance, "linked_training_with_distractor": eligible,
                        "rows": audit, "elapsed_seconds": perf_counter() - start})
        return

    state = torch.load(a.checkpoint, weights_only=True, map_location="cpu")
    assert all(state["provenance"][k] == provenance[k] for k in
               ("fixture_sha256", "config_sha256", "weights_sha256", "index_sha256", "models"))
    heads = {arm: (PairScorer() if arm == "matched_plain" else ContextScorer()).cuda().eval()
             for arm in FIT_ARMS}
    for arm, head in heads.items():
        head.load_state_dict(state["heads"][arm])
    reader = Reader(a.assets / "reader")
    rows, requests = [], []
    with torch.inference_mode():
        cases = test[a.shard::a.shards]
        queries = encoder([q["question"] for q in cases], "query")
        for i, case in enumerate(cases):
            query = queries[i:i + 1]
            ranked = index.search("dense", case["question"], query, depth)
            example, keys, gold, base, x, context, affinity, links = pool(case, ranked, query)
            gold_mask = torch.zeros(depth, device="cuda", dtype=torch.bool)
            gold_mask[list(gold)] = True
            hit = gold_mask[gpu_pairs[0]] & gold_mask[gpu_pairs[1]]
            gold_linked = bool((hit & links).any()) if len(gold) == 2 else None
            arms, _ = route(encoder, model, [example], query, keys, include_frozen=False)
            choices = {arm: pairs[0] for arm, pairs in arms.items()}
            logits = {"one_pass_pair": original(base, x)}
            logits.update({arm: head(base, x if arm == "matched_plain" else context)
                           for arm, head in heads.items()})
            link_ranking = {}
            for arm, logit in logits.items():
                choices[arm], linked_winner = selections(logit, gpu_pairs, affinity, links)
                if gold_linked and int(links.sum()) > 1:
                    gold_logit = logit[hit][0]
                    distractors = logit[links & ~hit]
                    link_ranking[arm] = {
                        "top": bool(hit[linked_winner]),
                        "pairwise_win": float(((distractors < gold_logit).float() +
                                               0.5 * (distractors == gold_logit).float()).mean()),
                    }
            records = {}
            for arm, picked in choices.items():
                left, right = sorted(picked)
                records[arm] = {"pair": picked, "selected_linked": bool((links & (gpu_pairs[0] == left) &
                                                                         (gpu_pairs[1] == right)).any()),
                                "both_support": len(gold) == 2 and set(gold) == set(picked)}
                requests.append((reader.prompt(example, picked), records[arm], case["answer"]))
            rows.append({"id": case["id"], "row_ids": ranked, "gold_slots": gold, "gold_linked": gold_linked,
                         "linked_pairs": int(links.sum()), "link_ranking": link_ranking, "arms": records})
        prompts = list(dict.fromkeys(p for p, _, _ in requests))
        outputs = dict(zip(prompts, reader.answers(prompts), strict=True))
        for prompt, record, answer in requests:
            record.update(prediction=outputs[prompt], **measure_answer(outputs[prompt], answer))
    save(a.output, {**provenance, "checkpoint_sha256": digest(a.checkpoint), "rows": rows,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "elapsed_seconds": perf_counter() - start})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "eval"))
    for name in ("assets", "inputs", "fixture", "output", "config", "weights", "checkpoint"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--training-cache", type=Path)
    p.add_argument("--title-cache", type=Path)
    p.add_argument("--baseline-sha256", help="verify released weights when the source caches are unavailable")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()
    if (args.training_cache is None) != (args.title_cache is None):
        p.error("provide both source caches together")
    if args.training_cache is None and args.baseline_sha256 is None:
        p.error("provide the source caches or --baseline-sha256")
    if not 0 <= args.shard < args.shards:
        p.error("shard must be in [0, shards)")
    run(args)


if __name__ == "__main__":
    main()
