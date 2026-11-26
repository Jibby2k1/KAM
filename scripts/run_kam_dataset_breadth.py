#!/usr/bin/env python3
"""Evaluate frozen HotpotQA selectors across supplied and independent news pools."""

import argparse
from itertools import chain
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest, words
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_fullwiki_support_links import ContextScorer, FIT_ARMS, relation_features, selections
from run_kam_hotpot_head_to_head import route
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pair_scoring import PairScorer
from run_kam_hotpot_pool_scaling import Reader, prepared
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, batches, candidate_texts, measure_answer


def padded_article_chunks(body):
    words_ = body.split()
    for start in range(0, len(words_), 96):
        yield " ".join(words_[start:start + 128])
        if start + 128 >= len(words_):
            break


def build_news(corpus, encoder):
    contexts, owners, bounds = [], [], []
    for i, row in enumerate(corpus):
        start = len(contexts)
        for chunk in padded_article_chunks(row["body"]):
            contexts.append((row["title"], [chunk]))
            owners.append(i)
        assert len(contexts) > start
        bounds.append((start, len(contexts)))
    bank = encoder(candidate_texts({"context": contexts}), "passage")
    return contexts, bank, torch.tensor(owners, device="cuda"), bounds


def news_pools(query, rows, corpus, built, depth):
    contexts, bank, owners, bounds = built
    # Stable article order breaks any exact score ties. Chunks are ranked
    # by affinity and the first chunk wins a tie within an article.
    raw = query @ bank.T
    pooled = torch.full((len(query), len(corpus)), -torch.inf, device="cuda")
    pooled.scatter_reduce_(1, owners.expand(len(query), -1), raw, reduce="amax")
    articles = pooled.argsort(dim=1, descending=True, stable=True)[:, :depth].tolist()
    url_ids = {row["url"]: i for i, row in enumerate(corpus)}
    for i, (row, ranking, scores) in enumerate(zip(rows, articles, raw, strict=True)):
        lo, hi = zip(*(bounds[n] for n in ranking), strict=True)
        chosen = [start + int(scores[start:end].argmax()) for start, end in zip(lo, hi, strict=True)]
        evidence = {url_ids[e["url"]]: e["fact"] for e in row["evidence"]}
        gold = [p for p, article in enumerate(ranking) if article in evidence]
        # A strict evidence-text audit in addition to the article-level labels.
        fact_visible = [p for p in gold if " ".join(words(evidence[ranking[p]])) in
                        " ".join(words(contexts[chosen[p]][1][0]))]
        example = {"question": row["question"], "context": [contexts[n] for n in chosen]}
        keys = bank[torch.tensor(chosen, device="cuda")].unsqueeze(0)
        yield i, example, keys, gold, {"article_ids": ranking, "chunk_ids": chosen, "visible_gold": fact_visible}


def supplied_pools(rows, enc):
    case_iter = ({"question": row["question"], "context": row["context"]} for row in rows)
    cases = list(case_iter)
    sizes = [len(case["context"]) for case in cases]
    bank = enc(list(chain.from_iterable(map(candidate_texts, cases))), "passage")
    for i, (row, example, keys) in enumerate(zip(rows, cases, bank.split(sizes), strict=True)):
        yield i, example, keys.unsqueeze(0), row["gold"], {}


def fullwiki_pools(rows, index):
    for i, row in enumerate(rows):
        ranks = row["row_ids"]
        keys = index.bank[torch.tensor(ranks, device="cuda")].unsqueeze(0)
        yield i, index.candidates(row, ranks), keys, row["gold_slots"], {}


@torch.inference_mode()
def evaluate(a, cfg, data, cases):
    enc, reader = Encoder(a.assets / "encoder", 64), Reader(a.assets / "reader")
    adapter, original = restore_checkpoint(a.baseline, a.config, cfg, None, None,
                                           weights_sha256=a.baseline_sha256)
    weights = torch.load(a.heads, weights_only=True, map_location="cpu")
    assert weights["provenance"]["weights_sha256"] == a.baseline_sha256
    assert weights["provenance"]["models"] == read(a.assets / "revisions.json")
    heads = {"one_pass_pair": original}
    for arm in FIT_ARMS:
        head = (PairScorer() if arm == "matched_plain" else ContextScorer()).cuda().eval()
        head.load_state_dict(weights["heads"][arm])
        heads[arm] = head
    ablation_masks = {}
    if a.ablations:
        fitted = torch.load(a.ablations, map_location="cpu", weights_only=True)
        assert fitted["baseline_sha256"] == a.baseline_sha256
        assert fitted["reference_heads_sha256"] == digest(a.heads)
        assert fitted["config_sha256"] == digest(a.config)
        for arm, item in fitted["arms"].items():
            head = ContextScorer().cuda().eval()
            head.load_state_dict(item["weights"])
            heads[arm] = head
            ablation_masks[arm] = torch.tensor(item["mask"], device="cuda")

    queries = enc([case["question"] for case in cases], "query")
    if data["dataset"] == "news":
        assert digest(a.corpus) == data["corpus_sha256"]
        corpus = read(a.corpus)
        pools = news_pools(queries, cases, corpus, build_news(corpus, enc), a.news_depth)
    elif data["dataset"] == "hotpot":
        pools = fullwiki_pools(cases, FullWikiIndex(a.inputs, read(a.inputs / "manifest.json"), cfg))
    else:
        pools = supplied_pools(cases, enc)
    records, requests = [], []
    pair_indices = {}
    for i, example, keys, gold, metadata in pools:
        query, row = queries[i:i + 1], cases[i]
        size = len(example["context"])
        if size not in pair_indices:
            cpu = torch.triu_indices(size, size, 1)
            pair_indices[size] = cpu, cpu.cuda()
        pairs, gpu_pairs = pair_indices[size]
        graph, lexical = relation_features(example, pairs)
        prepared_graph = graph.unsqueeze(0).cuda()
        base, x, affinity = prepared(adapter(query), keys, prepared_graph, gpu_pairs)
        mask = (graph[pairs[0], pairs[1]] | graph[pairs[1], pairs[0]]).cuda()
        both = len(gold) == 2
        gold_linked = bool(graph[gold[0], gold[1]] | graph[gold[1], gold[0]]) if both else None
        contest = gold_linked and int(mask.sum()) > 1
        routes, _ = route(enc, adapter, [example], query, keys, include_frozen=False)
        choices = {arm: rows[0] for arm, rows in routes.items()}
        diagnostic = {}
        augmented = torch.cat((x[0], lexical.cuda()), dim=-1)
        for arm, head in heads.items():
            if arm in ablation_masks:
                features = augmented * ablation_masks[arm]
            else:
                features = augmented if arm in ("context_only", "support_link") else x[0]
            logits = head(base[0], features)
            choices[arm], best_link = selections(logits, gpu_pairs, affinity[0], mask)
            if contest:
                diagnostic[arm] = set(pairs[:, best_link].tolist()) == set(gold)
        outcomes = {}
        for arm, pair in choices.items():
            left, right = pair
            evidence = both and set(pair) == set(gold)
            outcomes[arm] = {"pair": pair, "both_support": evidence,
                             "selected_linked": bool(graph[left, right] | graph[right, left])}
            if "visible_gold" in metadata:
                outcomes[arm]["both_evidence_spans"] = set(pair) == set(metadata["visible_gold"])
            requests.append((reader.prompt(example, pair), outcomes[arm], row))
        if both:
            oracle = sorted(gold, key=lambda p: (-float(affinity[0, p]), p))
            outcomes["oracle"] = {"pair": oracle, "both_support": True, "selected_linked": gold_linked}
            requests.append((reader.prompt(example, oracle), outcomes["oracle"], row))
        records.append({"id": row["id"], "kind": row["kind"], "gold": gold, "gold_linked": gold_linked,
                        "linked_pairs": int(mask.sum()), "ranking": diagnostic, "depth": size,
                        "arms": outcomes, **metadata})
    prompts = list(dict.fromkeys(prompt for prompt, _, _ in requests))
    prediction = dict(zip(prompts, reader.answers(prompts), strict=True))
    for prompt, item, row in requests:
        targets = [row["answer"], *row.get("aliases", [])]
        quality = [measure_answer(prediction[prompt], target) for target in targets]
        item["em"] = max(m["em"] for m in quality)
        item["f1"] = max(m["f1"] for m in quality)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("assets", "config", "fixture", "baseline", "heads", "output"):
        p.add_argument("--" + option, type=Path, required=True)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--ablations", type=Path)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--news-depth", type=int, default=64)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    if not 0 <= a.shard < a.shards:
        p.error("shard must be in [0, shards)")
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    cfg, data = read(a.config), read(a.fixture)
    cases = data["rows"][a.shard::a.shards]
    rows = evaluate(a, cfg, data, cases)
    assert [r["id"] for r in rows] == [q["id"] for q in cases]
    save(a.output, {"dataset": data["dataset"], "fixture_sha256": digest(a.fixture),
                    "ablations_sha256": digest(a.ablations) if a.ablations else None,
                    "baseline_sha256": a.baseline_sha256, "heads_sha256": digest(a.heads),
                    "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                    "elapsed_seconds": perf_counter() - start, "rows": rows})


if __name__ == "__main__":
    main()
