#!/usr/bin/env python3
"""Extract sentence features on disjoint, unmodified full-Wikipedia retrieval pools."""

import argparse
from hashlib import sha256
from pathlib import Path
import gzip
import json
from time import perf_counter

import numpy as np
import torch

from audit_kam_natural_data import digest, words
from prepare_kam_natural_router import title_key
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_fullwiki_support_links import ContextScorer, STOP, relation_features
from run_kam_hotpot_independent import read
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, measure_answer
from run_kam_sentence_verifier import COLUMNS, flush, rank_variant
from run_kam_sentence_verifier_fullwiki import pool_text


def reader_rows(reader, item, stage, example, affinity, answer, models, prompts):
    case = item["variants"]["clean"]
    scores = np.asarray(case["features"], dtype=np.float64)
    choices = {"frozen_context": case["winners"][0]}
    for name, model in models.items():
        x = (scores[:, model["columns"]] - model["mean"]) / model["scale"]
        choices[name] = int((x @ model["coefficients"]).argmax())
    arms = {}
    for name, idx in choices.items():
        pair = sorted(stage[3][idx], key=lambda j: (-affinity[j], j))
        record = {"both_support": idx == case["gold"]}
        prompts.setdefault(reader.prompt(example, pair), []).append((record, answer))
        arms[name] = record
    return {"id": item["id"], "arms": arms}


@torch.inference_mode()
def run(a):
    start = perf_counter()
    cfg, fixture, manifest = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json"))
    assert fixture["source_sha256"] == manifest["source_sha256"]
    assert fixture["training_depth"] == fixture["test_depth"] == 64
    names = [t for rows in fixture["rows"].values() for q in rows for t in q["gold_titles"]]
    assert len(names) == len(set(names))
    state = torch.load(a.heads, weights_only=True, map_location="cpu")
    ablated = torch.load(a.ablations, weights_only=True, map_location="cpu")
    assert ablated["reference_heads_sha256"] == digest(a.heads)
    assert state["provenance"]["models"] == read(a.assets / "revisions.json")
    assert state["provenance"]["index_sha256"] == digest(a.inputs / "lexical" / "completed.json")
    head, without = ContextScorer().cuda().eval(), ContextScorer().cuda().eval()
    head.load_state_dict(state["heads"]["context_only"])
    control = ablated["arms"]["refit_no_coverage"]
    without.load_state_dict(control["weights"])
    heads = head, without, torch.tensor(control["mask"], device="cuda")
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    index = FullWikiIndex(a.inputs, manifest, cfg)
    encoder = Encoder(a.assets / "encoder", 128)
    reader, models, prompts = None, {}, {}
    if a.fitted:
        reader = Reader(a.assets / "reader")
        fitted, transferred = read(a.fitted), read(a.old_policy)
        assert transferred["features"] == list(COLUMNS)
        models = {k: fitted[k] for k in ("native_base", "native_semantic", "native_linked_semantic")}
        models["transferred_2wiki"] = transferred | {"columns": list(range(len(COLUMNS)))}
    pool_rows = [{"split": split, **q} for split, cases in fixture["rows"].items() for q in cases]
    if a.test_only_reader_batches:
        pool_rows = [r for r in pool_rows if r["split"] == "test"]
    cases = pool_rows[a.shard::a.shards]
    # Repeat the extraction batch composition before skipping the fit rows.
    # Tied dense scores can change order across hardware or padding shapes.
    queries = encoder([q["question"] for q in cases], "query")
    pairs = torch.triu_indices(64, 64, 1)
    gpu_pairs = pairs.cuda()
    pending, metadata, reader_inputs, output = [], [], [], []

    def drain():
        features = flush(pending, encoder)
        if reader is not None:
            features = [reader_rows(reader, r, stage, example, affinity, answer, models, prompts)
                        for r, (stage, example, affinity, answer) in zip(features, reader_inputs, strict=True)]
            reader_inputs.clear()
        for r, details in zip(features, metadata, strict=True):
            output.append(r | details)
        pending.clear()
        metadata.clear()

    for case, query in zip(cases, queries.split(1), strict=True):
        if reader is not None and case["split"] != "test":
            continue
        ranked = index.search("dense", case["question"], query, 64)
        pool = pool_text(index.db, case["question"], ranked)
        gold_ids = set(case["gold_ids"]) - {None}
        gold = tuple(i for i, row_id in enumerate(ranked) if row_id in gold_ids)
        assert {ranked[i] for i, (title, _) in enumerate(pool["context"]) if title_key(title) in case["gold_titles"]} == set(ranked) & gold_ids
        graph, lexical = relation_features(pool, pairs)
        links = graph | graph.T
        linked_pairs = int(links.triu(1).sum())
        gold_linked = bool(links[gold[0], gold[1]]) if len(gold) == 2 else None
        bank = index.bank[torch.tensor(ranked, device="cuda")].unsqueeze(0)
        terms = set(words(case["question"])) - STOP
        context = {} if reader is not None else None
        stage = rank_variant("clean", pool, graph, lexical, bank, adapter(query), terms, heads, gpu_pairs, context)
        if reader is not None:
            example, affinity = context["clean"]
            reader_inputs.append((stage, example, affinity, case["answer"]))
        coverage = [len(set(pair).intersection(gold)) for pair in stage[3]]
        pending.append((case["id"], case["split"], gold, query[0], [stage]))
        metadata.append({"gold_linked": gold_linked, "gold_in_pool": len(gold), "linked_pairs": linked_pairs,
                         "shortlist_support_counts": coverage,
                         "pool_sha256": sha256(json.dumps(ranked, separators=(",", ":")).encode()).hexdigest()})
        if len(pending) == 16:
            drain()
    if pending:
        drain()
    if reader is not None:
        for prediction, entries in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
            for record, truth in entries:
                record["answer_exact"] = bool(measure_answer(prediction, truth)["em"])
    return {"fixture_sha256": digest(a.fixture), "index_sha256": digest(a.inputs / "lexical" / "completed.json"),
            "heads_sha256": digest(a.heads), "baseline_sha256": a.baseline_sha256,
            "ablations_sha256": digest(a.ablations), "columns": COLUMNS, "shard": a.shard, "shards": a.shards,
            "gpu": torch.cuda.get_device_name(), "seconds": perf_counter() - start, "rows": output,
            **({"fitted_sha256": digest(a.fitted), "old_policy_sha256": digest(a.old_policy)} if a.fitted else {})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "assets", "fixture", "config", "heads", "ablations", "baseline", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--fitted", type=Path)
    p.add_argument("--old-policy", type=Path)
    p.add_argument("--test-only-reader-batches", action="store_true")
    a = p.parse_args()
    if (a.fitted is None) != (a.old_policy is None):
        p.error("Use --fitted with --old-policy for the reader comparison")
    if a.test_only_reader_batches and a.fitted is None:
        p.error("--test-only-reader-batches requires the reader comparison")
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    data = json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode()
    a.output.write_bytes(gzip.compress(data, compresslevel=4, mtime=0))


if __name__ == "__main__":
    main()
