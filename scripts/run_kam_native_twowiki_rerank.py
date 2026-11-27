#!/usr/bin/env python3
"""Transfer a frozen native HotpotQA pair scorer to sealed 2Wiki document pools."""

import argparse
import gzip
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import torch

from audit_kam_natural_data import digest, words
from kam_union_relation_features import bitset_relation_features, induced_original_features
from run_kam_fullwiki_support_links import ContextScorer, STOP
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_transfer import restore_checkpoint
from run_kam_native_twowiki_reader import reference_rows
from run_kam_natural_feasibility import Encoder, measure_answer
from run_kam_sentence_verifier import COLUMNS, flush, rank_variant
from run_kam_sentence_verifier_fullwiki import SENTENCE_BREAK, selected_vectors

POOLS = ("dense64", "dense128", "conditioned128")


def prepared_pools(case, runs, db):
    dense, conditioned = (runs[k] for k in ("dense", "conditioned"))
    assert dense[:64] == conditioned[:64] and len(dense) == len(conditioned) == 128
    union = sorted(set(dense) | set(conditioned))
    marks = ",".join("?" for _ in union)
    looked_up = {i: (title, SENTENCE_BREAK.split(body)) for i, title, body in
                 db.execute(f"SELECT id, title, body FROM docs WHERE id IN ({marks})", union)}
    assert set(looked_up) == set(union)
    return {name: (rank, {"question": case["question"], "context": [looked_up[k] for k in rank]})
            for name, rank in zip(POOLS, (dense[:64], dense, conditioned), strict=True)}


def heads_for(a, index, revisions):
    state = torch.load(a.heads, map_location="cpu", weights_only=True)
    raw = torch.load(a.ablations, map_location="cpu", weights_only=True)
    assert state["provenance"]["index_sha256"] == index and state["provenance"]["models"] == revisions
    assert raw["reference_heads_sha256"] == digest(a.heads)
    full, control = ContextScorer().cuda().eval(), ContextScorer().cuda().eval()
    full.load_state_dict(state["heads"]["context_only"])
    ablation = raw["arms"]["refit_no_coverage"]
    control.load_state_dict(ablation["weights"])
    return full, control, torch.tensor(ablation["mask"], device="cuda")


@torch.inference_mode()
def run(a):
    start = perf_counter()
    fixture, config, manifest, revisions = (json.loads(p.read_text()) for p in
                                           (a.fixture, a.config, a.inputs / "manifest.json", a.assets / "revisions.json"))
    metadata = json.loads((a.inputs / "lexical/completed.json").read_text())
    assert all(metadata[k] == v for k, v in fixture["corpus"].items())
    assert fixture["source_sha256"] == digest(a.source) and digest(a.fitted) == a.fitted_sha256
    searches = [json.loads(gzip.decompress(p.read_bytes())) for p in a.search]
    assert sorted(p["shard"] for p in searches) == list(range(4))
    assert all(p["fixture_sha256"] == digest(a.fixture) and p["encoder"] == revisions and p["corpus"] == fixture["corpus"]
               for p in searches)
    for part in searches:
        assert [r["id"] for r in part["rows"]] == [r["id"] for r in fixture["rows"][part["shard"]::4]]
    cases = fixture["rows"][a.shard::a.shards]
    rankings = {r["id"]: r for item in searches for r in item["rows"]}
    refs = {} if a.features_only else reference_rows(a.source, {q["id"] for q in cases})
    wanted = sorted({d for q in cases for column in ("dense", "conditioned") for d in rankings[q["id"]][column]})
    embeddings, offsets = selected_vectors(a.inputs, manifest, config["encoding_parts_per_row_group"], wanted, "cuda")
    adapter, _ = restore_checkpoint(a.baseline, a.config, config, None, None, weights_sha256=a.baseline_sha256)
    heads = heads_for(a, digest(a.inputs / "lexical/completed.json"), revisions)
    locked = json.loads(a.fitted.read_text())["native_semantic"]
    assert locked["columns"] == list(range(len(COLUMNS)))
    encoder = Encoder(a.assets / "encoder", 128)
    reader = None if a.features_only else Reader(a.assets / "reader")
    pair_ids = {n: torch.triu_indices(n, n, 1) for n in (64, 128)}
    gpu_pairs = {n: p.cuda() for n, p in pair_ids.items()}
    base_pairs = pair_ids[128][1] < 64
    assert torch.equal(pair_ids[128][:, base_pairs], pair_ids[64])
    rows, reader_requests = [], []
    db_file = (a.inputs / "lexical/articles.sqlite").resolve()
    with sqlite3.connect(f"{db_file.as_uri()}?immutable=1", uri=True) as db:
        for offset in range(0, len(cases), 12):
            chunk = cases[offset:offset+12]
            query = encoder([q["question"] for q in chunk], "query")
            adapted = adapter(query)
            plans, context = [], []
            for i, case in enumerate(chunk):
                pools = prepared_pools(case, rankings[case["id"]], db)
                relations = {k: bitset_relation_features(pools[k][1], pair_ids[128]) for k in POOLS[1:]}
                relations["dense64"] = induced_original_features(relations["conditioned128"], 64, base_pairs)
                pipeline, bank = [], {}
                terms = set(words(case["question"])) - STOP
                for name, (rank, example) in pools.items():
                    graph, lexical = relations[name]
                    values = embeddings[torch.tensor([offsets[k] for k in rank], device="cuda")][None]
                    pipeline.append(rank_variant(name, example, graph, lexical, values, adapted[i:i+1],
                                                 terms, heads, gpu_pairs[len(rank)], bank))
                plans.append((case["id"], "test", (), query[i], pipeline))
                context.append((pools, bank))

            for case, plan, features, (pools, contexts) in zip(chunk, plans, flush(plans, encoder), context, strict=True):
                arms = {}
                gold = {u["corpus_id"] for u in case["units"]}
                for name, _, _, pairs, _, _ in plan[4]:
                    if a.features_only and name != "conditioned128":
                        continue
                    rank, example = pools[name]
                    values = features["variants"][name]["features"]
                    order = contexts[name][1]
                    if a.features_only:
                        proposals = [[rank[j] for j in sorted(pair, key=lambda k: (-order[k], k))] for pair in pairs]
                        gold_index = next((i for i, proposal in enumerate(proposals) if set(proposal) == gold), -1)
                        arms[name] = {"gold": gold_index, "pairs": proposals, "features": values}
                        continue
                    x = np.asarray(values, dtype=np.float64)
                    scores = ((x - locked["mean"]) / locked["scale"]) @ locked["coefficients"]
                    chosen = sorted(pairs[int(scores.argmax())], key=lambda j: (-order[j], j))
                    pair = [rank[j] for j in chosen]
                    shortlisted = any({rank[j], rank[k]} == gold for j, k in pairs)
                    arm = {"pair": pair, "shortlisted": shortlisted}
                    arms[name] = arm
                    reader_requests.append((reader.prompt(example, chosen), refs[case["id"]]["answer"], arm))
                rows.append({"id": case["id"], "arms": arms})
    if reader_requests:
        prompts = list(dict.fromkeys(p for p, _, _ in reader_requests))
        answers = dict(zip(prompts, reader.answers(prompts), strict=True))
        for prompt, answer, arm in reader_requests:
            arm.update(measure_answer(answers[prompt], answer))
    outputs = {"fixture_sha256": digest(a.fixture), "fitted_sha256": digest(a.fitted), "source_sha256": digest(a.source),
               "corpus": fixture["corpus"], "models": revisions, "search_hashes": sorted(map(digest, a.search)),
               "device": torch.cuda.get_device_name(), "shard": a.shard, "shards": a.shards,
               "elapsed_seconds": perf_counter() - start, "rows": rows}
    if a.features_only:
        outputs["columns"] = list(COLUMNS)
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("assets", "inputs", "config", "fixture", "source", "baseline", "heads", "ablations", "fitted", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    for key in ("baseline-sha256", "fitted-sha256"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--search", nargs=4, required=True, type=Path)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--features-only", action="store_true")
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    result = run(a)
    a.output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
