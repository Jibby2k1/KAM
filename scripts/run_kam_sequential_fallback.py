#!/usr/bin/env python3
"""Score frozen broad and sequential retrieval and log pre-hop routing features."""

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
from run_kam_fullwiki_support_links import ContextScorer, STOP, relation_features
from run_kam_hotpot_head_to_head import route
from run_kam_hotpot_independent import read
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, measure_answer
from run_kam_sentence_verifier import COLUMNS, flush, rank_variant
from run_kam_sentence_verifier_fullwiki import pool_text, selected_vectors

FEATURES = (
    "broad_pair_link", "pool_link_fraction", "top8_link_fraction", "broad_margin",
    "linked_minus_unlinked_margin", "broad_mean_affinity", "broad_min_affinity",
    "broad_weak_sentence_semantic", "broad_mention_margin", "initial_top_is_in_broad",
    "initial_margin", "question_terms",
)


def broad_choice(pair_features, stage, model, affinity, links):
    """Return the usual broad prediction and gate inputs before a conditioned query."""
    x = np.asarray(pair_features, dtype=np.float64)
    scores = ((x - model["mean"]) / model["scale"]) @ model["coefficients"]
    scores = (scores - scores.mean()) / max(scores.std(), 1e-8)
    winner = int(scores.argmax())
    chosen = sorted(stage[3][winner], key=lambda j: (-affinity[j], j))
    top = sorted(range(len(affinity)), key=lambda j: (-affinity[j], j))
    top_links = links[top[:8]][:, top[:8]]
    linked = x[:, 4].astype(bool)
    cross = scores[linked].max() - scores[~linked].max() if linked.any() and (~linked).any() else 0.
    features = [
        x[winner, 4], int(links.sum()) / (64 * 63), int(top_links.sum()) / (8 * 7),
        scores[winner] - np.partition(scores, -2)[-2], cross, x[winner, 3], x[winner, 2],
        x[winner, 11], x[winner, 15], top[0] in chosen, affinity[top[0]] - affinity[top[1]], len(stage[2]),
    ]
    return chosen, [round(float(v), 7) for v in features]


@torch.inference_mode()
def run(a):
    start = perf_counter()
    cfg, fixture, manifest = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json"))
    metadata = {"fixture_sha256": digest(a.fixture), "index_sha256": digest(a.inputs / "lexical" / "completed.json"),
                "shard": a.shard, "shards": a.shards, "gpu": torch.cuda.get_device_name()}
    sealed = json.loads(gzip.decompress(a.pools.read_bytes()))
    assert fixture["source_sha256"] == manifest["source_sha256"]
    assert all(sealed[k] == metadata[k] for k in ("fixture_sha256", "index_sha256"))
    assert fixture["training_depth"] == fixture["test_depth"] == 64
    groups = [{**q, "split": split} for split, rows in fixture["rows"].items() for q in rows]
    assert len(sealed["rows"]) == len(groups) and set(sealed["rows"]) == {q["id"] for q in groups}
    cases = groups[a.shard::a.shards]
    bank, by_id = selected_vectors(a.inputs, manifest, cfg["encoding_parts_per_row_group"],
                                   [i for q in cases for i in sealed["rows"][q["id"]]], "cuda")
    model = read(a.fitted)["native_semantic"]
    assert model["columns"] == list(range(len(COLUMNS)))
    state = torch.load(a.heads, weights_only=True, map_location="cpu")
    ablation = torch.load(a.ablations, weights_only=True, map_location="cpu")
    revisions = read(a.assets / "revisions.json")
    assert state["provenance"]["index_sha256"] == metadata["index_sha256"]
    assert state["provenance"]["models"] == revisions
    assert ablation["reference_heads_sha256"] == digest(a.heads)
    broad_head, without = ContextScorer().cuda().eval(), ContextScorer().cuda().eval()
    broad_head.load_state_dict(state["heads"]["context_only"])
    control = ablation["arms"]["refit_no_coverage"]
    without.load_state_dict(control["weights"])
    heads = broad_head, without, torch.tensor(control["mask"], device="cuda")
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    encoder = Encoder(a.assets / "encoder", 128)
    queries = encoder([q["question"] for q in cases], "query")
    reader = Reader(a.assets / "reader")
    pairs = torch.triu_indices(64, 64, 1)
    gpu_pairs = pairs.cuda()
    pending, details, prompts, output = [], [], {}, []

    def drain():
        visible = flush(pending, encoder)
        broad = [broad_choice(item["variants"]["clean"]["features"], stage, model, affinity, links)
                 for item, (_, stage, _, affinity, _, links, _) in zip(visible, details, strict=True)]
        sequential, _ = route(encoder, adapter, [d[2] for d in details],
                              torch.stack([p[3] for p in pending]), torch.cat([d[4] for d in details]),
                              include_frozen=False)
        for info, (first, features), second in zip(details, broad, sequential["learned_sequential"], strict=True):
            row, _, example, _, _, _, case = info
            ranked = sealed["rows"][case["id"]]
            arms = {}
            for name, selected in (("broad", first), ("sequential", second)):
                arms[name] = {"pair": selected, "support": {ranked[j] for j in selected} == set(case["gold_ids"])}
                prompts.setdefault(reader.prompt(example, selected), []).append((arms[name], case["answer"]))
            output.append(row | {"features": features, "arms": arms})
        pending.clear()
        details.clear()

    with sqlite3.connect(f"file:{a.inputs / 'lexical' / 'articles.sqlite'}?immutable=1", uri=True) as db:
        db.execute("PRAGMA mmap_size=1073741824")
        for case, query in zip(cases, queries.split(1), strict=True):
            id_ = case["id"]
            ranked = sealed["rows"][id_]
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
            details.append((row, stage, example, context["clean"][1], pool_keys, links, case))
            if len(pending) == 16:
                drain()
        if pending:
            drain()
    for predicted, requests in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
        for outcome, truth in requests:
            outcome["exact"] = bool(measure_answer(predicted, truth)["em"])
    return metadata | {"pools_sha256": digest(a.pools), "fitted_sha256": digest(a.fitted),
                       "heads_sha256": digest(a.heads), "ablations_sha256": digest(a.ablations),
                       "baseline_sha256": a.baseline_sha256, "models": revisions, "features": FEATURES,
                       "seconds": perf_counter() - start, "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "assets", "config", "fixture", "output", "baseline", "heads", "ablations", "pools", "fitted"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    a.output.write_bytes(gzip.compress(json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
