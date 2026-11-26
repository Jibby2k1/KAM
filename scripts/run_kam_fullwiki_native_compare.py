#!/usr/bin/env python3
"""Compare global BM25, dense, and concurrent hybrid retrieval on full Wikipedia."""

import argparse
from collections import defaultdict
from heapq import nsmallest
import json
from pathlib import Path
from random import Random
import sqlite3
from statistics import median
from time import perf_counter

# Load the PyTorch module runtime before the native index dependencies.
import torch
import numpy as np
import tantivy

from analyze_kam_pool_scaling import percentile
from audit_kam_natural_data import digest, words
from prepare_kam_natural_router import title_key
from run_kam_hotpot_head_to_head import adapter
from run_kam_hotpot_independent import METHODS, choose_options, read, save
from run_kam_hotpot_pair_scoring import links as original_links
from run_kam_hotpot_pool_scaling import Reader, fit_pair, pair_selection
from run_kam_hotpot_title_cache import lookup_links, sparse_index
from run_kam_natural_feasibility import Encoder, measure_answer


def reciprocal_rank_fusion(lexical, dense, k, rank_constant):
    fused = defaultdict(float)
    for items in (lexical, dense):
        for pos, row in enumerate(items, 1):
            fused[row] += 1 / (rank_constant + pos)
    return nsmallest(k, fused, key=lambda row: (-fused[row], row))


class FullWikiIndex:
    def __init__(self, directory, manifest, cfg, *, bank_device="cuda"):
        lexical = directory / "lexical"
        self.meta = read(lexical / "completed.json")
        assert self.meta["source_sha256"] == manifest["source_sha256"]
        assert self.meta["queries_sha256"] == manifest["queries_sha256"]
        assert self.meta["scanned"] == manifest["total_rows"]
        assert self.meta["mask_sha256"] == digest(lexical / "allowed.npy")
        mask = np.load(lexical / "allowed.npy", allow_pickle=False)
        assert mask.shape == (manifest["total_rows"],) and mask.dtype == np.bool_
        assert int(mask.sum()) == self.meta["indexed"]
        self.mask = torch.from_numpy(mask).cuda()
        self.zero_scores = np.flatnonzero(mask)[:2048].tolist()
        self.lexical_search_threads = cfg.get("lexical_search_threads")
        options = {} if self.lexical_search_threads is None else {"search_threads": self.lexical_search_threads}
        self.index = tantivy.Index.open(str(lexical / "tantivy"), **options)
        self.searcher = self.index.searcher()
        assert self.searcher.num_docs == self.meta["indexed"]
        self.lexical_pruning_df = cfg.get("lexical_pruning_df")
        if self.lexical_pruning_df is not None:
            assert 0 < self.lexical_pruning_df < 1 and self.meta["fast_row_id"]
        if self.lexical_search_threads is not None:
            assert self.lexical_search_threads > 0 and self.meta["fast_row_id"]
        self.db = sqlite3.connect(f"file:{lexical / 'articles.sqlite'}?immutable=1", uri=True)
        self.db.execute("PRAGMA mmap_size=1073741824")
        self.rank_constant = cfg["rrf_rank_constant"]
        self.bank = None
        parts = cfg["encoding_parts_per_row_group"]
        sizes = manifest["row_groups"]
        for shard in range(len(sizes) * parts):
            p = torch.load(directory / f"vectors_{shard:02d}.pt", weights_only=True, map_location="cpu")
            assert p["source_sha256"] == manifest["source_sha256"] and p["shard"] == shard
            group, part = shard % len(sizes), shard // len(sizes)
            offset = sum(sizes[:group]) + part * sizes[group] // parts
            end = sum(sizes[:group]) + (part + 1) * sizes[group] // parts
            vec = p["vectors"]
            assert p["offset"] == offset and len(vec) == end - offset and vec.dtype == torch.float32
            if self.bank is None:
                self.bank = torch.empty((manifest["total_rows"], vec.shape[1]), device=bank_device)
            self.bank[offset:end].copy_(vec)
        self.bank_half = self.bank.half() if cfg.get("dense_rescore") else None
        torch.cuda.synchronize()

    def lexical(self, question, k, *, engine=None, pruning_df=None):
        index, searcher = engine or (self.index, self.searcher)
        max_df = self.lexical_pruning_df if pruning_df is None else pruning_df
        if max_df is not None or self.lexical_search_threads is not None:
            from run_kam_fullwiki_lexical_pruning import cutoff_hits, lexical_terms, parse_terms, stable_ids

            terms = lexical_terms(question, searcher, max_df)
            query = parse_terms(index, terms)
            hits = cutoff_hits(searcher, query, k)
            ids = [row for row, _ in stable_ids(searcher, hits, k)]
        else:
            text = " ".join(f'"{token}"' for token in sorted(set(words(question))))
            query = index.parse_query(text, ["contents"])
            hits = searcher.search(query, limit=k, count=False).hits
            ids = [searcher.doc(address).get_first("row_id") for _, address in hits]
        remaining = set(ids)
        if len(ids) < k:
            ids.extend(i for i in self.zero_scores if i not in remaining and len(ids) < k)
        assert len(ids) == k and len(set(ids)) == k
        return ids

    def search(self, method, question, q, k):
        if method == "bm25":
            return self.lexical(question, k)
        if method == "dense" and self.bank_half is not None:
            scores = torch.mm(q.half(), self.bank_half.T, out_dtype=torch.float32)[0]
            shortlist = scores.masked_fill_(~self.mask, -torch.inf).topk(k + 32).indices
            vectors = self.bank[shortlist].unsqueeze(0)
            exact = torch.bmm(q.unsqueeze(1), vectors.transpose(1, 2)).flatten()
            return shortlist[exact.topk(k).indices].tolist()
        depth = k if method == "dense" else min(2 * k, 1024)
        # Queue dense scoring first so the independent CPU lexical search can overlap it.
        scores = self.bank @ q[0]
        ranked = scores.masked_fill_(~self.mask, -torch.inf).topk(depth).indices
        if method == "dense":
            return ranked.tolist()
        lex = self.lexical(question, depth)
        return reciprocal_rank_fusion(lex, ranked.tolist(), k, self.rank_constant)

    def candidates(self, q, ranked):
        places = ",".join("?" for _ in ranked)
        rows = {i: (title, [body]) for i, title, body in
                self.db.execute(f"SELECT id,title,body FROM docs WHERE id IN ({places})", ranked)}
        assert len(rows) == len(ranked)
        return {"question": q["question"], "context": [rows[i] for i in ranked]}


def local_title_links(example):
    refs = [("query", i) for i in range(len(example["context"]))]
    sparse = sparse_index({"query": example}, [refs])
    return lookup_links(example, refs, sparse)


def run(a):
    started = perf_counter()
    cfg = read(a.config)
    if cfg.get("lexical_search_threads") is not None:
        torch.set_float32_matmul_precision("highest")
    if cfg.get("dense_rescore"):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    manifest = read(a.inputs / "manifest.json")
    qpath = a.inputs / "queries.json"
    assert digest(qpath) == manifest["queries_sha256"]
    questions = read(qpath)
    cohort_sha = None
    if a.test_fixture:
        fixture = read(a.test_fixture)
        assert fixture["source_sha256"] == manifest["source_sha256"]
        assert fixture["previous_queries_sha256"] == manifest["queries_sha256"]
        original = [q for group in questions.values() for q in group]
        fresh = [{**q, "_id": q["id"]} for q in fixture["rows"]["test"]]
        assert not {q["_id"] for q in fresh} & {q["_id"] for q in original}
        assert len({t for q in fresh + original for t in q["gold_titles"]}) == 2 * (len(fresh) + len(original))
        questions["test"] = fresh
        cohort_sha = digest(a.test_fixture)
    index = FullWikiIndex(a.inputs, manifest, cfg)
    enc, reader = Encoder(a.assets / "encoder", 64), Reader(a.assets / "reader")
    model = adapter(a, cfg)
    mentions = torch.load(a.title_cache, map_location="cpu", weights_only=True)["train_mentions"].cuda()
    scorer = fit_pair(a, cfg, model, mentions, True)
    pair_indices = {k: torch.triu_indices(k, k, 1, device="cuda") for k in cfg["depths"]}
    setup_seconds = perf_counter() - started
    print(f"Loaded encoders and {index.meta['indexed']} indexed titles in {setup_seconds:.2f}s", flush=True)

    @torch.inference_mode()
    def request(q, method, depth, audit=False):
        torch.cuda.synchronize()
        start = perf_counter()
        query = enc([q["question"]], "query")
        torch.cuda.synchronize()
        search_start = perf_counter()
        selected = index.search(method, q["question"], query, depth)
        search_seconds = perf_counter() - search_start
        sample = index.candidates(q, selected)
        relation = local_title_links(sample)
        if audit:
            assert torch.equal(relation, original_links([sample])[0])
        keys = index.bank.index_select(0, torch.tensor(selected, device="cuda")).unsqueeze(0)
        chosen = pair_selection(model(query), keys, relation.unsqueeze(0).cuda(), scorer, pair_indices[depth])[0].tolist()
        retrieval_seconds = perf_counter() - start
        answer = reader.answers([reader.prompt(sample, chosen)])[0]
        torch.cuda.synchronize()
        end = perf_counter()
        return sample, selected, chosen, answer, (end - start, search_seconds, retrieval_seconds)

    rng = Random(cfg["seed"])
    depths_by_method = cfg.get("depths_by_method", dict.fromkeys(METHODS, cfg["depths"]))
    options = [(method, depth) for method in METHODS for depth in depths_by_method[method]]
    if a.frozen_options:
        selected = read(a.frozen_options)
        frozen_settings = [(point["method"], point["depth"]) for point in selected["chosen"]]
        assert len(frozen_settings) == len(METHODS) and {method for method, _ in frozen_settings} == set(METHODS)
        assert all(setting in options for setting in frozen_settings)
        warm_settings = frozen_settings
    else:
        warm_settings = options
    for setting in warm_settings:
        request(questions["calibration"][0], *setting, audit=True)
    if not a.frozen_options:
        samples = rng.sample(questions["calibration"][1:], cfg["calibration_timed_n"])
        durations = defaultdict(list)
        for q in samples:
            for option in rng.sample(options, len(options)):
                *_, elapsed = request(q, *option)
                durations[option].append(elapsed[0])
        timing = [{"method": method, "depth": depth, "p50_ms": 1000 * median(durations[method, depth]),
                   "p95_ms": 1000 * percentile(durations[method, depth], .95)}
                  for method, depth in options]
        calibration = {"timings": timing, "samples": [q["_id"] for q in samples]}
        if cfg.get("match_priority") == "latency_slack":
            for cell in timing:
                cell["sample_seconds"] = durations[cell["method"], cell["depth"]]
        save(a.output / "calibration_grid.json", calibration)
        selected = choose_options(timing, cfg)
    save(a.output / "operating_points.json", selected)
    print(f"Frozen whole-corpus budget {selected['budget_ms']}, depths "
          f"{[(point['method'], point['depth']) for point in selected['chosen']]}", flush=True)
    results = []
    for q in questions["test"]:
        for opt in rng.sample(list(selected["chosen"]), len(METHODS)):
            method, depth = opt["method"], opt["depth"]
            example, ranks, chosen, answer, secs = request(q, method, depth)
            titles = [title_key(t) for t, _ in example["context"]]
            gold = set(q["gold_titles"])
            assert len(set(titles)) == len(titles)
            supports = [titles[i] for i in chosen]
            results.append({"id": q["_id"], "method": method, "depth": depth,
                            "ranked_ids": ranks, "ranked_titles": titles,
                            "selected_titles": supports, "gold_in_pool": len(gold & set(titles)),
                            "both_support": gold <= set(supports), "prediction": answer,
                            **measure_answer(answer, q["answer"]),
                            "total_seconds": secs[0], "search_seconds": secs[1], "retrieval_seconds": secs[2]})
    save(a.output / "results.json", {
        "status": "completed", "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "tantivy": tantivy.__version__, "bank_resident_bytes": index.bank.nbytes,
        "half_bank_resident_bytes": index.bank_half.nbytes if index.bank_half is not None else 0,
        "setup_seconds": setup_seconds, "elapsed_seconds": perf_counter() - started,
        "source_sha256": manifest["source_sha256"], "queries_sha256": manifest["queries_sha256"],
        "test_fixture_sha256": cohort_sha,
        "frozen_options_sha256": digest(a.frozen_options) if a.frozen_options else None,
        "index_manifest_sha256": digest(a.inputs / "lexical/completed.json"),
        "lexical_pruning_df": index.lexical_pruning_df,
        "lexical_search_threads": index.lexical_search_threads,
        "config_sha256": digest(a.config), "models": read(a.assets / "revisions.json"),
        "records": results,
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "output", "config", "training-cache", "title-cache"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--test-fixture", type=Path)
    p.add_argument("--frozen-options", type=Path)
    p.set_defaults(shard=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
