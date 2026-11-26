#!/usr/bin/env python3
"""Evaluate a frozen sentence reranker on the original full-Wikipedia HotpotQA pools."""

import argparse
from pathlib import Path
import re
import sqlite3
from time import perf_counter

import torch

from audit_kam_natural_data import digest, words
from run_kam_fullwiki_support_links import ContextScorer, STOP, relation_features
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, measure_answer
from run_kam_sentence_verifier import COLUMNS, flush, rank_variant, reader_requests

SENTENCE_BREAK = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9\"“])")


def selected_vectors(root, manifest, parts, rows, device):
    wanted = set(rows)
    row_groups = manifest["row_groups"]
    shards = []
    offset = 0
    for group, span in enumerate(row_groups):
        for part in range(parts):
            low = offset + part * span // parts
            high = offset + (part + 1) * span // parts
            shards.append((low, high, part * len(row_groups) + group))
        offset += span
    index = sorted(wanted)
    banks, selected = [], []
    left = 0
    for low, high, shard in sorted(shards):
        right = left
        while right < len(index) and index[right] < high:
            right += 1
        if right > left:
            path = root / f"vectors_{shard:02d}.pt"
            data = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
            assert data["source_sha256"] == manifest["source_sha256"] and data["offset"] == low
            assert data["shard"] == shard and data["vectors"].dtype == torch.float32
            ids = index[left:right]
            banks.append(data["vectors"][[i - low for i in ids]])
            selected.extend(ids)
        left = right
    assert selected == index
    flat = torch.cat(banks).to(device)
    return flat, {v: i for i, v in enumerate(index)}


def pool_text(db, question, ranked):
    slots = ",".join("?" for _ in ranked)
    found = {i: (title, SENTENCE_BREAK.split(body)) for i, title, body in
             db.execute(f"SELECT id,title,body FROM docs WHERE id IN ({slots})", ranked)}
    assert len(found) == 64
    return {"question": question, "context": [found[i] for i in ranked]}


@torch.inference_mode()
def evaluate(a):
    device = "cuda"
    t0 = perf_counter()
    cfg, fixture, manifest, policy = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json", a.policy))
    assert policy["features"] == list(COLUMNS)
    assert fixture["source_sha256"] == manifest["source_sha256"]
    baseline_audit = {row["id"]: row for path in sorted(a.audit.glob("eval-*.json")) for row in read(path)["rows"]}
    test = fixture["rows"]["test"]
    assert len(test) == len(baseline_audit) == 768
    test = test[a.shard::a.shards]
    selected = [i for q in test for i in baseline_audit[q["id"]]["row_ids"]]
    vectors, by_id = selected_vectors(a.inputs, manifest, cfg["encoding_parts_per_row_group"], selected, device)
    heads_checkpoint = torch.load(a.heads, map_location="cpu", weights_only=True)
    ablation_checkpoint = torch.load(a.ablations, map_location="cpu", weights_only=True)
    assert heads_checkpoint["provenance"]["fixture_sha256"] == digest(a.fixture)
    assert heads_checkpoint["provenance"]["models"] == read(a.assets / "revisions.json")
    assert ablation_checkpoint["reference_heads_sha256"] == digest(a.heads)
    full, without = ContextScorer().to(device).eval(), ContextScorer().to(device).eval()
    full.load_state_dict(heads_checkpoint["heads"]["context_only"])
    ablation = ablation_checkpoint["arms"]["refit_no_coverage"]
    without.load_state_dict(ablation["weights"])
    heads = full, without, torch.tensor(ablation["mask"], device=device)
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    enc = Encoder(a.assets / "encoder", 128, device)
    reader = Reader(a.assets / "reader", device)
    queries = enc([q["question"] for q in test], "query")
    pairs = torch.triu_indices(64, 64, 1)
    gpu_pairs = pairs.to(device)
    db_path = a.inputs / "lexical" / "articles.sqlite"
    plans, contexts, prompts, rows = [], {}, {}, []

    def drain():
        if not plans:
            return
        features = flush(plans, enc)
        rows.extend(reader_requests(plans, features, contexts, policy, reader, prompts))
        plans.clear()

    with sqlite3.connect(f"file:{db_path}?immutable=1", uri=True) as db:
        db.execute("PRAGMA mmap_size=1073741824")
        for case, query in zip(test, queries.split(1), strict=True):
            id_ = case["id"]
            prev = baseline_audit[id_]
            ranked = prev["row_ids"]
            assert len(ranked) == 64 and len(set(ranked)) == 64
            pool = pool_text(db, case["question"], ranked)
            graph, lexical = relation_features(pool, pairs)
            gold = tuple(prev["gold_slots"])
            edge = bool(graph[gold[0], gold[1]] | graph[gold[1], gold[0]]) if len(gold) == 2 else None
            assert edge == prev["gold_linked"]
            assert set(ranked[i] for i in gold) == set(ranked).intersection(case["gold_ids"])
            assert int((graph | graph.T).triu(1).sum()) == prev["linked_pairs"]
            bank = vectors[[by_id[i] for i in ranked]].unsqueeze(0)
            keep = {}
            stage = rank_variant("clean", pool, graph, lexical, bank, adapter(query),
                                 set(words(case["question"])) - STOP, heads, gpu_pairs, keep)
            contexts[id_] = (case["answer"], keep)
            plans.append((id_, "hotpot-test", gold, query[0], [stage]))
            if len(plans) == 16:
                drain()
    drain()
    for prediction, entries in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
        for record, truth in entries:
            record["exact"] = bool(measure_answer(prediction, truth)["em"])
    for row in rows:
        prev = baseline_audit[row["id"]]
        row["gold_linked"] = prev["gold_linked"]
        row["linked_pairs"] = prev["linked_pairs"]
        full_record = row["variants"]["clean"]["full"]
        old = prev["arms"]["context_only"]
        row["matches_previous_evidence"] = full_record["evidence"] == old["both_support"]
        row["matches_previous_exact"] = full_record["exact"] == bool(old["em"])
    return {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "seconds": perf_counter() - t0,
            "policy_sha256": digest(a.policy), "fixture_sha256": digest(a.fixture),
            "models": read(a.assets / "revisions.json"), "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "assets", "config", "fixture", "audit", "baseline", "heads", "ablations", "policy", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    save(a.output, evaluate(a))


if __name__ == "__main__":
    main()
