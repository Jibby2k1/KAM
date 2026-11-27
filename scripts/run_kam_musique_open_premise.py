#!/usr/bin/env python3
"""Retrieve an earlier MuSiQue passage before fixed two-endpoint routing."""

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import tantivy
import torch

from audit_kam_natural_data import digest, stable, words
from prepare_kam_native_musique import canonical_paragraph
from prepare_kam_natural_router import title_key
from run_kam_fullwiki_native_compare import reciprocal_rank_fusion
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_endpoint_pairs import build_prompts, exact, route_prompts
from run_kam_musique_reader_composition import read_packed, tokenized_question, unpack
from run_kam_musique_reader_windows import READERS
from run_kam_native_musique import COMPONENT_DEPTH, corpus_and_fixture, lexical_rank
from run_kam_natural_feasibility import Encoder


def open_both(items, questions, ranked, encoder, bank, reader, articles, ordered_pairs, article_map):
    tokenizer = reader.tokenizer
    render = lambda i: articles[i]["title"] + ". " + articles[i]["body"]
    prior = tokenizer("Candidate passages:", add_special_tokens=False).input_ids
    separator = tokenizer(" ; ", add_special_tokens=False).input_ids
    selected = {}
    for family in ("dense", "hybrid", "bm25"):
        initial = ranked[family]
        first = [p[0] for p in initial]
        old_masks = torch.tensor([r[:128] for r in initial], device="cuda")
        text = tokenizer.batch_decode(tokenizer([render(i) for i in first], add_special_tokens=False,
                                                max_length=128, truncation=True).input_ids)
        update = encoder([q+" Previous evidence: "+p for q,p in zip(questions, text, strict=True)], "query")
        conditioned = update @ bank.T
        confirmed = conditioned.gather(1, old_masks).argsort(dim=-1, descending=True, stable=True)
        shared = old_masks.gather(1, confirmed).tolist()
        fresh = conditioned.argsort(dim=-1, descending=True, stable=True)[:, :192].tolist()
        selected[family] = {
            "direct": [(r[:128], r[:2]) for r in initial],
            "same_pool": [(r[:128], [a, next(p for p in q if p != a)]) for a, q, r in zip(first, shared, initial, strict=True)],
            "conditioned_search": [],
        }
        for original, proposals in zip(initial, fresh, strict=True):
            basis = original[:64]
            banned = set(basis)
            extension = [k for k in proposals if k not in banned][:64]
            anchor = basis[0]
            assert len(extension) == 64
            second = next(p for p in proposals if p != anchor)
            assert second in basis + extension
            selected[family]["conditioned_search"].append((basis + extension, [anchor, second]))
    example = []
    for item in items:
        row = item[0]
        q = next(q for q in ordered_pairs[row["pair"]] if q["id"] == row["question"])
        gold = [article_map[d["id"]] for d in q["evidence"]]
        example.append({k: row[k] for k in ("question", "family", "pair")} | {"annotated_route": gold, "systems": {}})
    for family, variants in selected.items():
        for name, proposals in variants.items():
            prompts = []
            for i, (_, route) in enumerate(proposals):
                q = tokenized_question(tokenizer, questions[i])
                tokens = [tokenizer(render(x), add_special_tokens=False, max_length=160, truncation=True).input_ids for x in route]
                for view in (tokens[0] + separator + tokens[1], tokens[1]):
                    ids = tuple(q + prior + view + [tokenizer.eos_token_id])
                    prompts.append((ids, (1,)*len(ids)))
            answers = read_packed(reader, prompts)
            for i, ((pool, route), row) in enumerate(zip(proposals, example, strict=True)):
                found = [row["annotated_route"][j] in pool for j in (0,1)]
                hit = [exact(x, items[i][1]) for x in answers[2*i:2*i+2]]
                row["systems"][family+"/"+name] = {"pool": pool, "selected": route, "contains_both": all(found),
                                                      "exact_answer_both": hit[0], "exact_answer_second_only": hit[1]}
    return example


def rerank_open(items, questions, question_vectors, encoder, bank, reader, articles, fixed):
    tokenizer = reader.tokenizer
    intro = tokenizer("Candidate passages:", add_special_tokens=False).input_ids
    rows = fixed["rows"]
    assert [i[0]["question"] for i in items] == [r["question"] for r in rows]
    controls = [row["systems"]["dense/conditioned_search"] for row in rows]
    anchors = [r["selected"][0] for r in controls]
    pools = [r["pool"] for r in controls]
    positions = torch.tensor(pools, dtype=torch.long, device="cuda")

    def text(i):
        return articles[i]["title"] + ". " + articles[i]["body"]

    descriptions = tokenizer.batch_decode(tokenizer([text(i) for i in anchors], add_special_tokens=False,
                                                      max_length=128, truncation=True).input_ids)
    update = encoder([q+" Previous evidence: "+doc for q,doc in zip(questions,descriptions,strict=True)], "query")
    reference = (update @ bank.T).gather(1, positions)
    original = (question_vectors @ bank.T).gather(1, positions)
    similarity = (bank[anchors] @ bank.T).gather(1, positions)
    ignore = positions.eq(torch.tensor(anchors, device="cuda")[:, None])
    baseline = reference.masked_fill(ignore, -torch.inf)
    order = baseline.argsort(dim=-1, descending=True, stable=True)
    in_top_eight = torch.zeros_like(ignore).scatter_(1, order[:, :8], True)
    mentions = []
    for passage, pool in zip(descriptions, pools, strict=True):
        body = " " + " ".join(words(passage)) + " "
        titles = [words(articles[i]["title"]) for i in pool]
        mentions.append([len(tokens) >= 2 and (" " + " ".join(tokens) + " ") in body for tokens in titles])
    mentioned = torch.tensor(mentions, dtype=torch.bool, device="cuda") & ~ignore
    temperature = (reference.amax(1, keepdim=True) - reference.amin(1, keepdim=True) + 1)
    policies = {
        "conditioned": reference, "question_blend": .75*reference + .25*original,
        "subtract_anchor": reference - .25*similarity,
        "favor_mentioned": reference + temperature*mentioned,
        "favor_mentioned_top_eight": reference + temperature*(mentioned & in_top_eight),
    }
    manifest = [{k: r[k] for k in ("question", "family", "pair", "annotated_route")} | {
        "anchor": c["selected"][0], "pool": c["pool"], "results": {}} for r,c in zip(rows,controls,strict=True)]
    for mode, scores in policies.items():
        slots = scores.masked_fill(ignore, -torch.inf).argmax(-1, keepdim=True)
        choices = positions.gather(1, slots)[:, 0].tolist()
        if mode == "conditioned":
            assert choices == [r["selected"][1] for r in controls]
        prompts = []
        for q, second in zip(questions, choices, strict=True):
            desc = tokenizer(text(second), add_special_tokens=False, max_length=160, truncation=True).input_ids
            ids = tuple(tokenized_question(tokenizer, q) + intro + desc + [tokenizer.eos_token_id])
            prompts.append((ids, (1,)*len(ids)))
        answer = read_packed(reader, prompts)
        for i, (row, first, second, raw) in enumerate(zip(manifest, anchors, choices, answer, strict=True)):
            answer_exact = exact(raw, items[i][1])
            if mode == "conditioned":
                assert answer_exact == controls[i]["exact_answer_second_only"]
            row["results"][mode] = {"selected": [first, second], "answer_exact": answer_exact,
                                   "mentioned_second": bool(mentioned[i, pools[i].index(second)])}
    return manifest


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
    provenance = {"fixture_sha256": digest(a.fixture), "prior_sha256": digest(a.previous), "corpus_sha256": native["corpus_sha256"],
                  "models": models, "source_sha256": fixture["source_sha256"], "device": torch.cuda.get_device_name(), "shard": a.shard}
    if a.fully_open:
        outcomes = open_both(items, questions, ranked, encoder, bank, reader, articles, ordered_pairs, article_map)
        return provenance | {"rows": outcomes}
    if a.rerank_frozen:
        fixed = unpack(a.rerank_frozen)
        assert all(fixed[k] == provenance[k] for k in ("fixture_sha256", "corpus_sha256", "models", "shard"))
        revised = rerank_open(items, questions, answers, encoder, bank, reader, articles, fixed)
        return provenance | {"open_reference_sha256": digest(a.rerank_frozen), "rows": revised}
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
    return provenance | {"rows": exposed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "native-fixture", "fixture", "previous", "lexical", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--vectors", type=Path, nargs=4, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--fully-open", action="store_true")
    p.add_argument("--rerank-frozen", type=Path)
    a = p.parse_args()
    out = run(a)
    a.output.write_bytes(gzip.compress(json.dumps(out, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
