#!/usr/bin/env python3
"""Retrieve an earlier MuSiQue passage before fixed two-endpoint routing."""

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import tantivy
import torch

from audit_kam_natural_data import digest, stable
from prepare_kam_native_musique import canonical_paragraph
from prepare_kam_natural_router import title_key
from run_kam_fullwiki_native_compare import reciprocal_rank_fusion
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_endpoint_pairs import build_prompts, exact, route_prompts
from run_kam_musique_reader_composition import read_packed, unpack
from run_kam_musique_reader_windows import READERS
from run_kam_native_musique import COMPONENT_DEPTH, corpus_and_fixture, lexical_rank
from run_kam_natural_feasibility import Encoder


@torch.inference_mode()
def run(a):
    torch.set_float32_matmul_precision("highest")
    articles, native = corpus_and_fixture(argparse.Namespace(corpus=a.corpus, fixture=a.native_fixture))
    fixture = unpack(a.fixture)
    assert fixture["source_sha256"] == native["source_sha256"]
    previous = unpack(a.previous)
    assert previous["fixture_sha256"] == digest(a.fixture) and previous["shard"] == a.shard
    models = unpack(a.assets/"revisions.json")
    assert models == previous["model_revisions"]
    path = a.assets/f"reader_large_{READERS['large'][0]}"
    assert digest(path/"model.safetensors") == READERS["large"][1]
    reader = Reader(path)
    tokenizer = reader.tokenizer
    encoder = Encoder(a.assets/"encoder", 256)
    items, parts, _, rejected = build_prompts(fixture, a.shard, 4, tokenizer)
    assert rejected == previous["rejected"]
    selection = [(x, p) for x, p in zip(items, parts, strict=True) if x[0]["hops"] == 2 and not x[0]["reverse"]]
    items, parts = [list(x) for x in zip(*selection, strict=True)]
    ordered_pairs = {p["id"]: p["questions"] for p in fixture["pairs"]}

    vectors = sorted([torch.load(p, weights_only=True, map_location="cpu") for p in a.vectors], key=lambda x: x["begin"])
    assert all(p["corpus_sha256"] == native["corpus_sha256"] and p["models"] == models for p in vectors)
    assert vectors[0]["begin"] == 0 and vectors[-1]["end"] == len(articles)
    assert all(p["end"] == q["begin"] for p, q in zip(vectors, vectors[1:]))
    bank = torch.cat([r["vectors"] for r in vectors], 0).cuda()
    src = unpack(a.lexical/"meta.json")
    assert src["corpus_sha256"] == native["corpus_sha256"]
    index = tantivy.Index.open(str(a.lexical/"tantivy"))
    searcher = index.searcher()
    assert searcher.num_docs == len(bank) == len(articles)
    article_map = {stable("\x1f".join(canonical_paragraph({"title": r["title"], "paragraph_text": r["body"]}))): r["id"] for r in articles}
    assert len(article_map) == len(articles)
    titles = [title_key(r["title"]) for r in articles]
    first_gold, forbidden = [], []
    for item in items:
        row = item[0]
        pair = ordered_pairs[row["pair"]]
        q = next(q for q in pair if q["id"] == row["question"])
        first_gold.append(article_map[q["evidence"][0]["id"]])
        forbidden.append({title_key(k["evidence"][-1]["title"]) for k in pair})
    assert all(titles[i] not in exclusions for i, exclusions in zip(first_gold, forbidden, strict=True))
    questions = [x[3]["endpoints"] for x in items]
    answers = encoder(questions, "query")
    sem = (answers @ bank.T).argsort(dim=-1, descending=True, stable=True)[:, :COMPONENT_DEPTH].tolist()
    lex = [lexical_rank(index, searcher, q) for q in questions]
    ranked = {"dense": sem, "bm25": lex, "hybrid": [reciprocal_rank_fusion(a_, b, 128, 60) for a_, b in zip(lex, sem, strict=True)]}
    doc_tags = [x for item in items for x in item[4]]
    doc_features = encoder(doc_tags, "passage").view(len(items), 2, -1)
    visible_truth = [item[-1] for item in items]
    assert all(x in (0, 1) for x in visible_truth)
    earlier = {name: [[doc for doc in order if titles[doc] not in exclude][:8]
                      for order, exclude in zip(ranks, forbidden, strict=True)] for name, ranks in ranked.items()}
    assert all(len(x) == 8 for docs in earlier.values() for x in docs)
    original = {r["question"]: r for r in previous["rows"] if not r["reverse"] and r["hops"] == 2}
    assert set(original) == {item[0]["question"] for item in items}
    exposed = []
    for item, gold in zip(items, first_gold, strict=True):
        row = item[0]
        observed = {k: v.copy() for k, v in original[row["question"]].items() if k in ("dense_endpoints", "routed")}
        expose = {k: row[k] for k in ("question", "pair", "family")}
        exposed.append({**expose, "gold_earlier": gold,
                        "archived_joint_oracles": {k: bool(v["support_exact"] and v["exact"])
                                                   for k, v in observed["routed"]["large"].items()}, "open_routes": {}})
    for mode, nominees in earlier.items():
        for width in (1, 4):
            start = perf_counter()
            passage = [articles[d]["title"] + ". " + articles[d]["body"] for docs in nominees for d in docs[:width]]
            text = tokenizer.batch_decode(tokenizer(passage, add_special_tokens=False, max_length=128, truncation=True).input_ids)
            inquiries = [questions[j//width] + " Previous evidence: " + phrase for j, phrase in enumerate(text)]
            updates = encoder(inquiries, "query").view(len(items), width, -1)
            score = (updates[:, :, None, :] * doc_features[:, None, :, :]).sum(-1)
            # A larger absolute score margin selects a first passage on this fixed beam.
            chosen_prefix = (score[..., 0] - score[..., 1]).abs().argmax(1)
            row_ids = torch.arange(len(items), device="cuda")
            prediction = score[row_ids, chosen_prefix].argmax(1).tolist()
            prefix_slots = chosen_prefix.tolist()
            replies = read_packed(reader, route_prompts(parts, prediction, tokenizer, include_prefix=False))
            torch.cuda.synchronize()
            duration = perf_counter()-start
            for i, (row, choice, answer, gold, trail, slot) in enumerate(zip(exposed, prediction, replies, first_gold, nominees, prefix_slots, strict=True)):
                row["open_routes"][f"{mode}/{width}"] = {"prefix_candidates": trail, "premise_rank": trail.index(gold)+1 if gold in trail else None,
                                                         "selected_premise": trail[slot], "selected_endpoint": choice,
                                                         "premise_correct": trail[slot] == gold, "endpoint_correct": choice == visible_truth[i],
                                                         "answer_exact": exact(answer, items[i][1]), "foil_answer": exact(answer, items[i][2])}
            print(mode, width, duration, flush=True)
    return {"fixture_sha256": digest(a.fixture), "prior_sha256": digest(a.previous), "corpus_sha256": native["corpus_sha256"],
            "models": models, "source_sha256": fixture["source_sha256"], "device": torch.cuda.get_device_name(),
            "shard": a.shard, "rows": exposed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "native-fixture", "fixture", "previous", "lexical", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--vectors", type=Path, nargs=4, required=True)
    p.add_argument("--shard", type=int, required=True)
    a = p.parse_args()
    out = run(a)
    a.output.write_bytes(gzip.compress(json.dumps(out, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
