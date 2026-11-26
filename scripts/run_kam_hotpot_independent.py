#!/usr/bin/env python3
"""Benchmark BM25, exact dense and reciprocal-rank fusion at a calibrated latency budget."""

import argparse
from collections import Counter, defaultdict
from heapq import nsmallest
from itertools import product
import json
import math
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import numpy as np
import torch

from analyze_kam_pool_scaling import percentile
from audit_kam_natural_data import digest, words
from prepare_kam_natural_router import title_key
from run_kam_hotpot_head_to_head import adapter
from run_kam_hotpot_pool_scaling import Reader, fit_pair, pair_selection
from run_kam_hotpot_title_cache import lookup_links, sparse_index
from run_kam_natural_feasibility import Encoder, measure_answer

METHODS = ("bm25", "dense", "hybrid")


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def encode(a, docs, cfg, manifest):
    started = perf_counter()
    ids = list(range(len(docs)))[a.shard::cfg["encoding_shards"]]
    # Group similarly sized passages so GPU batches do less padding.
    ids.sort(key=lambda i: min(192, sum(len(s.split()) for s in docs[i][1])))
    encoder = Encoder(a.assets / "encoder", cfg["encoding_batch"], a.device)
    embeddings = encoder([docs[i][0] + ". " + " ".join(docs[i][1]) for i in ids], "passage").cpu()
    torch.save({"ids": ids, "vectors": embeddings, "corpus_sha256": manifest["corpus_sha256"],
                "device": torch.cuda.get_device_name() if a.device == "cuda" else "CPU",
                "elapsed_seconds": perf_counter() - started}, a.output / f"encode_{a.shard}.pt")
    print(f"Encoded {len(ids)} corpus documents in {perf_counter() - started:.2f} seconds", flush=True)


class Index:
    def __init__(self, docs, embeddings, device, fusion_constant):
        self.docs = [(title, sentences) for title, sentences, _ in docs]
        self.titles = [title_key(title) for title, _ in self.docs]
        self.bank = embeddings.to(device)
        self.fusion_constant = fusion_constant
        raw = defaultdict(list)
        lengths = []
        for i, (title, sentences) in enumerate(self.docs):
            terms = words(title + " " + " ".join(sentences))
            lengths.append(len(terms))
            for token, count in Counter(terms).items():
                raw[token].append((i, count))
        sizes = np.asarray(lengths)
        norm = 1.2 * (.25 + .75 * sizes / sizes.mean())
        self.postings = {}
        for token, row in raw.items():
            ids, tfs = np.asarray(row, dtype=np.int32).T
            idf = math.log1p((len(docs) - len(row) + .5) / (len(row) + .5))
            self.postings[token] = ids, (idf * 2.2 * tfs / (tfs + norm[ids])).astype(np.float32)
        self.links = sparse_index({"corpus": {"context": self.docs}},
                                  [[("corpus", i) for i in range(len(docs))]])

    def lexical(self, question, k):
        scores = np.zeros(len(self.docs), dtype=np.float32)
        for word in sorted(set(words(question))):
            posting = self.postings.get(word)
            if posting is not None:
                ids, weights = posting
                scores[ids] += weights
        kth = len(scores) - k
        cut = np.partition(scores, kth)[kth]
        winner = np.flatnonzero(scores > cut)
        winner = np.concatenate((winner, np.flatnonzero(scores == cut)[:k - len(winner)]))
        return winner[np.lexsort((winner, -scores[winner]))].tolist()

    def search(self, method, question, query, depth):
        if method == "bm25":
            return self.lexical(question, depth)
        k = depth if method == "dense" else min(depth * 2, 768)
        dense = torch.topk(self.bank @ query[0], k).indices.tolist()
        if method == "dense":
            return dense
        fused = defaultdict(float)
        for winners in (self.lexical(question, k), dense):
            for rank_, slot in enumerate(winners, 1):
                fused[slot] += 1.0 / (self.fusion_constant + rank_)
        return nsmallest(depth, fused, key=lambda i: (-fused[i], i))


def choose_options(timings, cfg):
    pick_ = {}
    for cap in cfg["budget_grid_ms"]:
        feasible = [[x for x in timings if x["method"] == arm and x["p50_ms"] <= cap[0]
                     and x["p95_ms"] <= cap[1]] for arm in METHODS]
        if not all(feasible):
            continue
        for combination in product(*feasible):
            p50 = [x["p50_ms"] for x in combination]
            p95 = [x["p95_ms"] for x in combination]
            spread50, spread95 = max(p50) - min(p50), max(p95) - min(p95)
            limit50, limit95 = cfg["match_spread_ms"]
            if spread50 > limit50 or spread95 > limit95:
                continue
            depths = [x["depth"] for x in combination]
            key = (min(depths), sum(depths), -spread50, -spread95)
            if cfg.get("match_priority") == "latency_slack":
                max_fraction = max(spread50 / limit50, spread95 / limit95, max(p50) / cap[0], max(p95) / cap[1])
                key = (-max_fraction, -spread50 / limit50 - spread95 / limit95, *key)
            if not pick_ or key > pick_["key"]:
                pick_ = {"key": key, "budget_ms": cap, "matched_on_calibration": True, "chosen": combination}
        if pick_:
            return pick_
    for cap in cfg["budget_grid_ms"]:
        feasible = [[x for x in timings if x["method"] == arm and x["p50_ms"] <= cap[0]
                     and x["p95_ms"] <= cap[1]] for arm in METHODS]
        if all(feasible):
            return {"budget_ms": cap, "matched_on_calibration": False,
                    "chosen": [max(f, key=lambda x: x["depth"]) for f in feasible]}
    raise ValueError("no common calibration budget; retain raw timings for review")


def evaluate(a, docs, cfg, manifest, questions):
    started = perf_counter()
    width = None
    seen = set()
    for i in range(cfg["encoding_shards"]):
        piece = torch.load(a.output / f"encode_{i}.pt", weights_only=True, map_location="cpu")
        assert piece["corpus_sha256"] == manifest["corpus_sha256"]
        assert set(piece["ids"]) == set(range(i, len(docs), cfg["encoding_shards"]))
        vectors = piece["vectors"]
        if width is None:
            width = vectors.shape[-1]
            bank = torch.empty((len(docs), width))
        bank[piece["ids"]] = vectors
        seen.update(piece["ids"])
    assert seen == set(range(len(docs)))
    build_started = perf_counter()
    index = Index(docs, bank, a.device, cfg["rrf_rank_constant"])
    index_build_seconds = perf_counter() - build_started
    encoder, reader = Encoder(a.assets / "encoder", 64, a.device), Reader(a.assets / "reader", a.device)
    model = adapter(a, cfg, a.device)
    mentions = torch.load(a.title_cache, map_location="cpu", weights_only=True)["train_mentions"].to(a.device)
    scorer = fit_pair(a, cfg, model, mentions, True)
    pairs = {depth: torch.triu_indices(depth, depth, 1, device=a.device) for depth in cfg["depths"]}
    title_set = set(index.titles)

    @torch.inference_mode()
    def request(q, method, depth):
        if a.device == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        query = encoder([q["question"]], "query")
        if a.device == "cuda":
            torch.cuda.synchronize()
        start_search = perf_counter()
        ranked = index.search(method, q["question"], query, depth)
        search_seconds = perf_counter() - start_search
        docs_ = [index.docs[i] for i in ranked]
        example = {"question": q["question"], "context": docs_}
        keys = index.bank.index_select(0, torch.tensor(ranked, device=a.device)).unsqueeze(0)
        relations = lookup_links(example, [("corpus", i) for i in ranked], index.links).unsqueeze(0).to(a.device)
        chosen = pair_selection(model(query), keys, relations, scorer, pairs[depth])[0].tolist()
        after_routing = perf_counter()
        answer = reader.answers([reader.prompt(example, chosen)])[0]
        if a.device == "cuda":
            torch.cuda.synchronize()
        elapsed = perf_counter() - start
        return ranked, [ranked[i] for i in chosen], answer, (elapsed, search_seconds, after_routing - start)

    rng = Random(cfg["seed"])
    options = list(product(METHODS, cfg["depths"]))
    warm = questions["calibration"][0]
    for setting in options:
        request(warm, *setting)
    calibration = rng.sample(questions["calibration"][1:], cfg["calibration_timed_n"])
    durations = defaultdict(list)
    for q in calibration:
        for setting in rng.sample(options, len(options)):
            _, _, _, times = request(q, *setting)
            durations[setting].append(times[0])
    timings = [{"method": method, "depth": depth, "p50_ms": 1000 * median(durations[method, depth]),
                "p95_ms": 1000 * percentile(durations[method, depth], .95)}
               for method, depth in options]
    save(a.output / "calibration_grid.json", {"timings": timings, "samples": [q["_id"] for q in calibration]})
    selected = choose_options(timings, cfg)
    save(a.output / "operating_points.json", selected)
    print(f"Frozen latency-matched options: {selected}", flush=True)
    records = []
    for group in ("test", "covered_test"):
        for q in questions[group]:
            for opt in rng.sample(list(selected["chosen"]), len(METHODS)):
                ranked, picked, answer, times = request(q, opt["method"], opt["depth"])
                gold = set(q["gold_titles"])
                found = [index.titles[i] for i in ranked]
                choices = [index.titles[i] for i in picked]
                records.append({"id": q["_id"], "group": group, "method": opt["method"], "depth": opt["depth"],
                                "gold_in_corpus": len(gold & title_set), "gold_in_pool": len(gold & set(found)),
                                "both_support": gold <= set(choices), "selected_titles": choices, "ranked_ids": ranked,
                                "prediction": answer, **measure_answer(answer, q["answer"]),
                                "total_seconds": times[0], "search_seconds": times[1], "retrieval_seconds": times[2]})
    save(a.output / "results.json", {"status": "completed", "corpus_sha256": manifest["corpus_sha256"],
                                    "questions_sha256": manifest["questions_sha256"], "config_sha256": digest(a.config),
                                    "torch": torch.__version__, "device": torch.cuda.get_device_name() if a.device == "cuda" else "CPU",
                                    "models": read(a.assets / "revisions.json"), "index_build_seconds": index_build_seconds,
                                    "elapsed_seconds": perf_counter() - started, "records": records})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("encode", "evaluate"))
    for name in ("assets", "dataset", "config", "training-cache", "title-cache", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = p.parse_args()
    docs_path, ques_path = a.dataset / "corpus.json", a.dataset / "questions.json"
    manifest = read(a.dataset / "manifest.json")
    assert (digest(docs_path), digest(ques_path)) == (manifest["corpus_sha256"], manifest["questions_sha256"])
    docs, cfg = read(docs_path), read(a.config)
    if a.mode == "encode":
        encode(a, docs, cfg, manifest)
    else:
        evaluate(a, docs, cfg, manifest, read(ques_path))


if __name__ == "__main__":
    main()
