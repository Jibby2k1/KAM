#!/usr/bin/env python3
"""Rerank frozen pools and their corpus-expanded unions with the same broad policy."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from audit_kam_natural_data import digest, words
from prepare_kam_natural_router import title_key
from run_kam_conditioned_corpus_search import load_gzip, top_corpus
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_fullwiki_support_links import ContextScorer, STOP, relation_features
from run_kam_hotpot_independent import read
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts, measure_answer
from run_kam_natural_router import score
from run_kam_sentence_verifier import COLUMNS, flush, rank_variant
from run_kam_sentence_verifier_fullwiki import pool_text, SENTENCE_BREAK

EXPANDED = ("unconditioned_union", "conditioned_union")
BATCH = 12


def union_example(index, example, original, top):
    original_ids = set(original)
    extra = [doc for doc in top if doc not in original_ids]
    extension = index.candidates(example, extra)["context"] if extra else []
    context = example["context"] + [(title, SENTENCE_BREAK.split(body)) for title, (body,) in extension]
    return original + extra, {"question": example["question"], "context": context}


@torch.inference_mode()
def run(a):
    started = perf_counter()
    cfg, fixture, manifest = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json"))
    sealed = load_gzip(a.pools)
    index_sha = digest(a.inputs / "lexical" / "completed.json")
    assert sealed["fixture_sha256"] == digest(a.fixture) and sealed["index_sha256"] == index_sha
    assert fixture["source_sha256"] == manifest["source_sha256"]
    cases = fixture["rows"]["test"]
    assert len(sealed["rows"]) == len(cases) == 1024 and set(sealed["rows"]) == {q["id"] for q in cases}
    cases = cases[a.shard::a.shards]
    revisions = read(a.assets / "revisions.json")
    state = torch.load(a.heads, weights_only=True, map_location="cpu")
    ablation = torch.load(a.ablations, weights_only=True, map_location="cpu")
    assert state["provenance"]["index_sha256"] == index_sha and state["provenance"]["models"] == revisions
    assert ablation["reference_heads_sha256"] == digest(a.heads)
    context_head, control_head = ContextScorer().cuda().eval(), ContextScorer().cuda().eval()
    context_head.load_state_dict(state["heads"]["context_only"])
    control = ablation["arms"]["refit_no_coverage"]
    control_head.load_state_dict(control["weights"])
    heads = context_head, control_head, torch.tensor(control["mask"], device="cuda")
    model = read(a.fitted)["native_semantic"]
    assert model["columns"] == list(range(len(COLUMNS)))
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    encoder = Encoder(a.assets / "encoder", 128)
    reader = Reader(a.assets / "reader")
    index = FullWikiIndex(a.inputs, manifest, cfg)
    pair_indices, prompts, output = {}, {}, []

    for offset in range(0, len(cases), BATCH):
        chunk = cases[offset:offset+BATCH]
        original = [sealed["rows"][c["id"]] for c in chunk]
        examples = [pool_text(index.db, q["question"], ids) for q, ids in zip(chunk, original, strict=True)]
        query = encoder([c["question"] for c in chunk], "query")
        adapted = adapter(query)
        ids = torch.tensor(original, device="cuda")
        first_slots = score(adapted, index.bank[ids]).argmax(dim=-1, keepdim=True)
        first_ids = ids.gather(1, first_slots).flatten().tolist()
        augmented = [f"{c['question']} First retrieved document: {candidate_texts(c)[slot]}"
                     for c, slot in zip(examples, first_slots.flatten().tolist(), strict=True)]
        second = adapter(encoder(augmented, "query"))
        candidates, _ = top_corpus(index, torch.cat((adapted, (adapted + second) * .5)), first_ids * 2)

        plans, details = [], []
        for i, (case, example, base) in enumerate(zip(chunk, examples, original, strict=True)):
            gold = set(case["gold_ids"])
            assert ({base[j] for j, (title, _) in enumerate(example["context"])
                     if title_key(title) in case["gold_titles"]} == gold.intersection(base))
            id_ = case["id"]
            hits = dict(zip(EXPANDED, (candidates[i], candidates[i + len(chunk)]), strict=True))
            variants = {"broad": (base, example)} | {
                arm: union_example(index, example, base, top) for arm, top in hits.items()
            }
            prepared, contexts = [], {}
            terms = set(words(case["question"])) - STOP
            for arm, (ranked, variant) in variants.items():
                n = len(ranked)
                if n not in pair_indices:
                    cpu_pairs = torch.triu_indices(n, n, 1)
                    pair_indices[n] = cpu_pairs, cpu_pairs.cuda()
                cpu_pairs, gpu_pairs = pair_indices[n]
                graph, lexical = relation_features(variant, cpu_pairs)
                bank = index.bank[torch.tensor(ranked, device="cuda")].unsqueeze(0)
                stage = rank_variant(arm, variant, graph, lexical, bank, adapted[i:i+1], terms, heads, gpu_pairs,
                                     contexts)
                prepared.append(stage)
            plans.append((id_, "test", (), query[i], prepared))
            details.append((case, base, variants, contexts, hits, first_ids[i]))

        scored = flush(plans, encoder)
        for item, plan, detail in zip(scored, plans, details, strict=True):
            case, base, variants, contexts, hits, first = detail
            gold = set(case["gold_ids"])
            arms = {}
            for stage in plan[4]:
                arm = stage[0]
                ranked, example = variants[arm]
                pairs = stage[3]
                x = np.asarray(item["variants"][arm]["features"], dtype=np.float64)
                scores = ((x - model["mean"]) / model["scale"]) @ model["coefficients"]
                order = contexts[arm][1]
                chosen = sorted(pairs[int(scores.argmax())], key=lambda j: (-order[j], j))
                shortlisted = any({ranked[j], ranked[k]} == gold for j, k in pairs)
                arms[arm] = {"pair": [ranked[j] for j in chosen], "gold_pair_in_shortlist": shortlisted}
                prompts.setdefault(reader.prompt(example, chosen), []).append((arms[arm], case["answer"]))
            output.append({"id": case["id"], "first": first, "corpus_hits": hits, "arms": arms,
                           "pool_hash": sha256(json.dumps(base, separators=(",", ":")).encode()).hexdigest()})

    for predicted, requests in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
        for arm, truth in requests:
            arm["exact"] = bool(measure_answer(predicted, truth)["em"])
    return {"fixture_sha256": digest(a.fixture), "pools_sha256": digest(a.pools), "config_sha256": digest(a.config),
            "index_sha256": index_sha, "fitted_sha256": digest(a.fitted), "heads_sha256": digest(a.heads),
            "ablations_sha256": digest(a.ablations), "baseline_sha256": a.baseline_sha256, "models": revisions,
            "indexed_documents": index.meta["indexed"], "shard": a.shard, "shards": a.shards,
            "gpu": torch.cuda.get_device_name(), "seconds": perf_counter() - started, "rows": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "config", "fixture", "pools", "fitted", "baseline", "heads", "ablations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    a = parser.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    payload = json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode()
    a.output.write_bytes(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    main()
