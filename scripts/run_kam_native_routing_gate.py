#!/usr/bin/env python3
"""Freeze Wikipedia pools, then score prefit routing specialists and the frozen reader."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
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
from run_kam_sentence_verifier_fullwiki import pool_text, selected_vectors

FEATURES = (
    "broad_pair_link", "linked_pair_link", "broad_margin", "linked_margin",
    "broad_cross_preference", "linked_cross_preference", "shared_document", "pool_link_fraction",
    "broad_mean_affinity", "linked_mean_affinity", "broad_weak_semantic", "linked_weak_semantic",
    "broad_mention_margin", "linked_mention_margin",
)
MODELS = {"broad": "native_semantic", "linked": "native_linked_semantic"}


def infer_gate_inputs(pair_features, pairs, fitted, graph_fraction):
    """Use only shortlist features, scorer outputs, graph density and selected pair identities."""
    x = np.asarray(pair_features, dtype=np.float64)
    logits, winners = {}, {}
    for name, key in MODELS.items():
        model = fitted[key]
        expected = list(range(len(COLUMNS)))
        assert model["columns"] == expected
        scores = ((x - model["mean"]) / model["scale"]) @ model["coefficients"]
        scores = (scores - scores.mean()) / max(scores.std(), 1e-8)
        logits[name] = scores
        winners[name] = int(scores.argmax())
    b, l = (winners[k] for k in MODELS)
    margins = [float(np.partition(s, -2)[-1] - np.partition(s, -2)[-2]) for s in logits.values()]
    features = [
        x[b, 4], x[l, 4], *margins, logits["broad"][b] - logits["broad"][l],
        logits["linked"][l] - logits["linked"][b], int(bool(set(pairs[b]) & set(pairs[l]))),
        graph_fraction, x[b, 3], x[l, 3], x[b, 11], x[l, 11], x[b, 15], x[l, 15],
    ]
    return winners, [round(float(v), 7) for v in features]


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


@torch.inference_mode()
def run(a):
    t0 = perf_counter()
    cfg, fixture, manifest = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json"))
    assert fixture["source_sha256"] == manifest["source_sha256"]
    assert fixture["training_depth"] == fixture["test_depth"] == 64
    groups = [{"split": split, **q} for split, items in fixture["rows"].items() for q in items]
    assert len({q["id"] for q in groups}) == len(groups)
    cases = groups[a.shard::a.shards]
    encoder = Encoder(a.assets / "encoder", 128)
    queries = encoder([q["question"] for q in cases], "query")
    metadata = {"fixture_sha256": digest(a.fixture), "index_sha256": digest(a.inputs / "lexical" / "completed.json"),
                "shard": a.shard, "shards": a.shards, "gpu": torch.cuda.get_device_name()}
    if a.mode == "freeze":
        index = FullWikiIndex(a.inputs, manifest, cfg)
        rows = [{"id": q["id"], "row_ids": index.search("dense", q["question"], vector, 64)}
                for q, vector in zip(cases, queries.split(1), strict=True)]
        return metadata | {"seconds": perf_counter() - t0, "rows": rows}

    frozen = unpack(a.pools)
    assert frozen["fixture_sha256"] == metadata["fixture_sha256"]
    assert frozen["index_sha256"] == metadata["index_sha256"]
    assert set(frozen["rows"]) == {q["id"] for q in groups}
    chosen_rows = [i for q in cases for i in frozen["rows"][q["id"]]]
    bank, by_id = selected_vectors(a.inputs, manifest, cfg["encoding_parts_per_row_group"], chosen_rows, "cuda")
    fitted = read(a.fitted)
    state = torch.load(a.heads, weights_only=True, map_location="cpu")
    no_coverage = torch.load(a.ablations, weights_only=True, map_location="cpu")
    assert state["provenance"]["index_sha256"] == metadata["index_sha256"]
    assert state["provenance"]["models"] == read(a.assets / "revisions.json")
    assert no_coverage["reference_heads_sha256"] == digest(a.heads)
    head, without = ContextScorer().cuda().eval(), ContextScorer().cuda().eval()
    head.load_state_dict(state["heads"]["context_only"])
    control = no_coverage["arms"]["refit_no_coverage"]
    without.load_state_dict(control["weights"])
    heads = head, without, torch.tensor(control["mask"], device="cuda")
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    reader = Reader(a.assets / "reader")
    pairs = torch.triu_indices(64, 64, 1)
    gpu_pairs = pairs.cuda()
    pending, details, prompts, output = [], [], {}, []

    def drain():
        for item, info in zip(flush(pending, encoder), details, strict=True):
            row, stage, example, affinity, case = info
            x = item["variants"]["clean"]
            choices, visible = infer_gate_inputs(x["features"], stage[3], fitted, row["pool_link_count"] / 2016)
            agree = choices["broad"] == choices["linked"]
            choices["frozen_context"] = x["winners"][0]
            outcomes = {}
            for name, idx in choices.items():
                pair = sorted(stage[3][idx], key=lambda j: (-affinity[j], j))
                outcome = {"support": idx == x["gold"], "pair": pair}
                if case["split"] == "test":
                    prompts.setdefault(reader.prompt(example, pair), []).append((outcome, case["answer"]))
                outcomes[name] = outcome
            output.append(row | {"features": visible, "agree": agree, "arms": outcomes})
        pending.clear()
        details.clear()

    with sqlite3.connect(f"file:{a.inputs / 'lexical' / 'articles.sqlite'}?immutable=1", uri=True) as db:
        db.execute("PRAGMA mmap_size=1073741824")
        for case, query in zip(cases, queries.split(1), strict=True):
            id_ = case["id"]
            ranked = frozen["rows"][id_]
            assert len(ranked) == len(set(ranked)) == 64
            example = pool_text(db, case["question"], ranked)
            gold_ids = set(case["gold_ids"]) - {None}
            gold = tuple(j for j, i in enumerate(ranked) if i in gold_ids)
            assert {ranked[i] for i, (title, _) in enumerate(example["context"])
                    if title_key(title) in case["gold_titles"]} == set(ranked) & gold_ids
            graph, lexical = relation_features(example, pairs)
            links = graph | graph.T
            linked = bool(links[gold[0], gold[1]]) if len(gold) == 2 else None
            pool_keys = bank[[by_id[i] for i in ranked]].unsqueeze(0)
            context = {}
            stage = rank_variant("clean", example, graph, lexical, pool_keys, adapter(query),
                                 set(words(case["question"])) - STOP, heads, gpu_pairs, context)
            pending.append((id_, case["split"], gold, query[0], [stage]))
            row = {"id": id_, "split": case["split"], "gold_in_pool": len(gold), "gold_linked": linked,
                   "pool_link_count": int(links.triu(1).sum()),
                   "pool_hash": sha256(json.dumps(ranked, separators=(",", ":")).encode()).hexdigest()}
            details.append((row, stage, example, context["clean"][1], case))
            if len(pending) == 16:
                drain()
        if pending:
            drain()
    for predicted, requests in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
        for outcome, truth in requests:
            outcome["exact"] = bool(measure_answer(predicted, truth)["em"])
    return metadata | {"pools_sha256": digest(a.pools), "fitted_sha256": digest(a.fitted), "heads_sha256": digest(a.heads),
                       "ablations_sha256": digest(a.ablations), "models": read(a.assets / "revisions.json"),
                       "features": FEATURES, "seconds": perf_counter() - t0, "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "score"))
    for key in ("inputs", "assets", "config", "fixture", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("baseline", "heads", "ablations", "pools", "fitted"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--baseline-sha256")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    if a.mode == "score" and not all((a.baseline, a.heads, a.ablations, a.pools, a.fitted, a.baseline_sha256)):
        p.error("Scoring requires a frozen pool artifact and the frozen checkpoints")
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    a.output.write_bytes(gzip.compress(json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
