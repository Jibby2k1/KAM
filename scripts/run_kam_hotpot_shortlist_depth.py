#!/usr/bin/env python3
"""Replay nested pair shortlist depths on the same sealed hybrid candidate unions."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from audit_kam_natural_data import digest, words
from run_kam_conditioned_corpus_search import load_gzip
from run_kam_conditioned_union import union_example
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_fullwiki_support_links import ContextScorer, STOP, relation_features
from run_kam_musique_transfer import restore_checkpoint
from run_kam_hotpot_independent import read
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import Encoder, measure_answer
from run_kam_sentence_verifier import COLUMNS, flush, rank_variant
from run_kam_sentence_verifier_fullwiki import pool_text

DEPTHS = (16, 32, 64)
BATCH = 12
METADATA = ("fixture_sha256", "pools_sha256", "config_sha256", "index_sha256", "fitted_sha256",
            "heads_sha256", "ablations_sha256", "baseline_sha256", "models", "indexed_documents")


@torch.inference_mode()
def run(a):
    started = perf_counter()
    cfg, fixture, manifest = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json"))
    sealed = load_gzip(a.pools)
    reference = load_gzip(a.reference)
    expected = {r["id"]: r for s in reference for r in s["rows"]}
    cases = fixture["rows"]["test"]
    ids = {r["id"] for r in cases}
    assert len(cases) == len(ids) == 1024 and expected.keys() == sealed["rows"].keys() == ids
    assert sealed["method"] == "hybrid" and sealed["pool_depth"] == 64
    index_sha = digest(a.inputs / "lexical" / "completed.json")
    assert fixture["source_sha256"] == manifest["source_sha256"]
    revisions = read(a.assets / "revisions.json")
    provenance = {name + "_sha256": digest(getattr(a, name))
                  for name in ("fixture", "pools", "config", "fitted", "heads", "ablations")}
    provenance |= {"index_sha256": index_sha, "models": revisions, "baseline_sha256": digest(a.baseline),
                   "indexed_documents": reference[0]["indexed_documents"]}
    assert all(all(s[k] == provenance[k] for k in METADATA) for s in reference)
    assert provenance["baseline_sha256"] == a.baseline_sha256 and sealed["index_sha256"] == index_sha
    cases = cases[a.shard::a.shards]

    index = FullWikiIndex(a.inputs, manifest, cfg)
    assert index.meta["indexed"] == provenance["indexed_documents"]
    state = torch.load(a.heads, weights_only=True, map_location="cpu")
    ablation = torch.load(a.ablations, weights_only=True, map_location="cpu")
    assert state["provenance"]["index_sha256"] == index_sha and state["provenance"]["models"] == revisions
    assert ablation["reference_heads_sha256"] == provenance["heads_sha256"]
    first_head, second_head = ContextScorer().cuda().eval(), ContextScorer().cuda().eval()
    first_head.load_state_dict(state["heads"]["context_only"])
    control = ablation["arms"]["refit_no_coverage"]
    second_head.load_state_dict(control["weights"])
    heads = first_head, second_head, torch.tensor(control["mask"], device="cuda")
    model = read(a.fitted)["native_semantic"]
    assert model["columns"] == list(range(len(COLUMNS)))
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    encoder, reader = Encoder(a.assets / "encoder", 128), Reader(a.assets / "reader")

    prompts, output, index_pairs = {}, [], {}
    for offset in range(0, len(cases), BATCH):
        chunk = cases[offset:offset + BATCH]
        query = encoder([q["question"] for q in chunk], "query")
        adapted = adapter(query)
        plans, details = [], []
        for i, case in enumerate(chunk):
            id_ = case["id"]
            base, old = sealed["rows"][id_], expected[id_]
            hits = old["corpus_hits"]["conditioned_union"]
            pool_hash = sha256(json.dumps(base, separators=(",", ":")).encode()).hexdigest()
            assert old["pool_hash"] == pool_hash and old["first"] in base
            ranked, example = union_example(index, pool_text(index.db, case["question"], base), base, hits)
            n = len(ranked)
            if n not in index_pairs:
                pair_ids = torch.triu_indices(n, n, 1)
                index_pairs[n] = pair_ids, pair_ids.cuda()
            cpu_pairs, gpu_pairs = index_pairs[n]
            graph, lexical = relation_features(example, cpu_pairs)
            bank = index.bank[torch.tensor(ranked, device="cuda")].unsqueeze(0)
            ctx = {}
            variant = rank_variant("expanded", example, graph, lexical, bank, adapted[i:i + 1],
                                   set(words(case["question"])) - STOP, heads, gpu_pairs, ctx, shortlist=DEPTHS[-1])
            plans.append((id_, "test", (), query[i], [variant]))
            details.append((ranked, example, ctx["expanded"][1], pool_hash))

        features = flush(plans, encoder)
        for case, plan, item, detail in zip(chunk, plans, features, details, strict=True):
            ranked, example, affinity, pool_hash = detail
            _, bank, _, pairs, columns, _ = plan[4][0]
            values = np.asarray(item["variants"]["expanded"]["features"], dtype=np.float64)
            score = ((values - model["mean"]) / model["scale"]) @ model["coefficients"]
            order = np.argsort(-np.asarray(columns)[:, :2], axis=0, kind="stable")
            gold = set(case["gold_ids"])
            arms = {}
            for depth in DEPTHS:
                subset = np.unique(order[:depth].flatten())
                top = pairs[int(subset[score[subset].argmax()])]
                oriented = sorted(top, key=lambda j: (-affinity[j], j))
                shortlisted = any({ranked[x] for x in pairs[j]} == gold for j in subset)
                documents = {d for j in subset for d in pairs[j]}
                sentences = {sentence for d in documents for sentence in bank[d][1]}
                arm = {"pair": [ranked[j] for j in oriented], "gold_pair_in_shortlist": shortlisted,
                       "candidate_pairs": len(subset), "candidate_documents": len(documents),
                       "distinct_sentences": len(sentences)}
                arms[str(depth)] = arm
                prompts.setdefault(reader.prompt(example, oriented), []).append((arm, case["answer"]))
            output.append({"id": case["id"], "pool_hash": pool_hash, "expanded_size": len(ranked), "arms": arms})

    for prediction, callbacks in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
        for arm, target in callbacks:
            arm["exact"] = bool(measure_answer(prediction, target)["em"])
    return {**provenance, "reference_sha256": digest(a.reference), "gpu": torch.cuda.get_device_name(),
            "shard": a.shard, "shards": a.shards, "seconds": perf_counter() - started,
            "pair_depths_per_head": DEPTHS, "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("inputs", "assets", "config", "baseline", "heads", "ablations", "fixture",
                  "pools", "fitted", "reference", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    data = json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode()
    a.output.write_bytes(gzip.compress(data, mtime=0))


if __name__ == "__main__":
    main()
