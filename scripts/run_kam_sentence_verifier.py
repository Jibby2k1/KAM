#!/usr/bin/env python3
"""Extract held-out sentence evidence for a shortlist of candidate support pairs."""

import argparse
from collections import Counter
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import torch
import numpy as np

from audit_kam_natural_data import digest, stable, words
from run_kam_fullwiki_support_links import ContextScorer, STOP, relation_features
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader, prepared
from run_kam_link_noise import challenge, modified
from run_kam_matched_pools import donors_by_quality
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts, measure_answer

SHORTLIST = 16
SCENARIOS = ("clean", "sham", "linked")
BASE = ("context_logit", "no_coverage_logit", "weak_affinity", "mean_affinity", "pair_link")
SENTENCE = ("weak_max_lexical", "max_lexical", "joint_lexical", "weak_first_sentence_lexical",
            "mention_lexical", "mention_same_sentence_count")
DENSE = ("weak_max_semantic", "max_semantic", "joint_top_semantic_lexical",
         "mention_semantic", "mention_margin")
COLUMNS = BASE + SENTENCE + DENSE


def sentence_bank(context, *, selected=None, cached=None, counts=None):
    result = [None] * len(context)
    positions = range(len(context)) if selected is None else sorted(selected)
    if counts is not None:
        counts["candidate_documents"] += len(context)
    for i in positions:
        title, sentences = context[i]
        key = None if cached is None else (title, tuple(sentences))
        if cached is not None and key in cached:
            result[i] = cached[key]
            continue
        entries = []
        remaining = 120
        for raw in islice(sentences, 8):
            tokens = words(raw)[:remaining]
            if tokens:
                entries.append(tuple(tokens))
            remaining -= len(tokens)
            if remaining <= 0:
                break
        if not entries:
            entries.append(())
        result[i] = tuple(words(title)), entries
        if cached is not None:
            cached[key] = result[i]
        if counts is not None:
            counts["normalized_documents"] += 1
    return result


def contains(tokens, phrase):
    return bool(phrase) and any(tokens[i:i+len(phrase)] == phrase for i, token in enumerate(tokens)
                                if token == phrase[0])


class DocumentTerms(NamedTuple):
    title: tuple
    title_words: set
    sentences: list
    overlaps: list
    semantic: list
    best_lexical: int
    best_semantic: int


def document_terms(question, bank, similarities, i):
    title, sentences = bank[i]
    overlaps = [question.intersection(sentence) for sentence in sentences]
    semantic = similarities[i]
    best_lexical = max(range(len(overlaps)), key=lambda j: len(overlaps[j]))
    best_semantic = max(range(len(semantic)), key=semantic.__getitem__)
    return DocumentTerms(title, set(title), sentences, overlaps, semantic, best_lexical, best_semantic)


def sentence_columns(question, bank, scores, pair, *, document_cache=None):
    norms = max(len(question), 1)
    docs = [document_terms(question, bank, scores, i) if document_cache is None else document_cache[i] for i in pair]
    left, right = docs
    lexical_max = [len(doc.overlaps[doc.best_lexical]) for doc in docs]
    max_semantic = [doc.semantic[doc.best_semantic] for doc in docs]
    cross = [(k, i) for k, doc in enumerate(docs) for i, tokens in enumerate(doc.sentences)
             if contains(tokens, docs[1-k].title)]
    cross_lex = max((len(docs[k].overlaps[i] - docs[1-k].title_words) for k, i in cross), default=0)
    cross_semantic = max((docs[k].semantic[i] for k, i in cross), default=0.)
    cross_gap = max((docs[k].semantic[i] - max_semantic[k] for k, i in cross), default=0.)
    return [
        min(lexical_max) / norms,
        max(lexical_max) / norms,
        len(left.overlaps[left.best_lexical] | right.overlaps[right.best_lexical]) / norms,
        min(len(left.overlaps[0]), len(right.overlaps[0])) / norms,
        cross_lex / norms,
        min(len(cross), 3),
        min(max_semantic),
        max(max_semantic),
        len(left.overlaps[left.best_semantic] | right.overlaps[right.best_semantic]) / norms,
        cross_semantic,
        cross_gap,
    ]


def plan_case(row, split, query, supplied, donor_keys, donors, donor_words, adapter, enc, heads, pairs,
              *, contextual_test=False, reader_context=None):
    adapted = adapter(query)
    proposals, _ = donors_by_quality(query, donor_keys, donor_words, row)
    items = [-(j + 1) for j in range(10)] + proposals["hard"]
    items.sort(key=lambda j: stable(f"kam-matched-order-v1:{row['id']}:{j}"))
    case = {"question": row["question"],
            "context": [row["context"][-x - 1] if x < 0 else donors[x] for x in items]}
    keys = torch.stack([supplied[-x - 1] if x < 0 else donor_keys[x] for x in items]).unsqueeze(0)
    gold = sorted(j for j, x in enumerate(items) if x < 0 and (-x - 1) in row["gold"])
    assert len(gold) == 2
    case["gold"] = gold
    cpu_pairs = pairs.cpu()
    graph, lexical = relation_features(case, cpu_pairs)
    _, _, affinity = prepared(adapted, keys, graph.unsqueeze(0).to(query.device), pairs)
    edit = challenge(case, graph, affinity[0])
    variants = [] if contextual_test else [("clean", case, graph, lexical, keys)]
    query_terms = set(words(row["question"])) - STOP
    if edit:
        donor, target, title, _ = edit
        # Preserve the previous matched scenario exactly for a direct paired comparison.
        fake = " ".join(f"registry{i+724}" for i, _ in enumerate(words(title)))
        template = "Additional references include {title}." if split.endswith("-fit") else "Related reading: {title}."
        modes = SCENARIOS[1:]
        if contextual_test:
            fragments = [words(s)[:18] for s in case["context"][donor][1][:8]]
            fragment = max(fragments, key=lambda tokens: len(query_terms.intersection(tokens)), default=[])
            template = " ".join(fragment) + "; for background, compare {title}."
            modes = ("contextual_sham", "contextual_linked")
        sham, linked = [modified(case, donor, label, template) for label in (fake, title)]
        changed = [relation_features(variant, cpu_pairs) for variant in (sham, linked)]
        if not changed[0][0][donor, target] and changed[1][0][donor, target]:
            assert all(sham["context"][g] == linked["context"][g] == case["context"][g] for g in gold)
            assert len(words(sham["context"][donor][1][0])) == len(words(linked["context"][donor][1][0]))
            edits = [candidate_texts({"context": [v["context"][donor]]})[0] for v in (sham, linked)]
            for mode, v, (g, x), vector in zip(modes, (sham, linked), changed, enc(edits, "passage"), strict=True):
                copy = keys.clone()
                copy[0, donor] = vector
                variants.append((mode, v, g, x, copy))

    result = [rank_variant(mode, variant, g, x, bank, adapted, query_terms, heads, pairs, reader_context)
              for mode, variant, g, x, bank in variants]
    return row["id"], split, tuple(gold), query[0], result


def rank_variant(mode, variant, graph, lexical, bank, adapted, query_terms, heads, pairs, reader_context=None,
                 *, shortlist=SHORTLIST, sentence_cache=None, sentence_counts=None):
    given, no_coverage, active = heads
    base, common, affinity = prepared(adapted, bank, graph.unsqueeze(0).to(adapted.device), pairs)
    inputs = torch.cat((common[0], lexical.to(adapted.device)), -1)
    scores = torch.stack((given(base[0], inputs), no_coverage(base[0], inputs * active)))
    scores = (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True).clamp_min(1e-6)
    ranks = scores.argsort(dim=1, descending=True, stable=True)[:, :shortlist]
    indices = ranks.flatten().unique(sorted=True)
    selected = pairs[:, indices].T
    lookup = {value: i for i, value in enumerate(indices.tolist())}
    winners = [lookup[j] for j in ranks[:, 0].tolist()]
    if reader_context is not None:
        reader_context[mode] = (variant, affinity[0].tolist())
    affinity = affinity[0, selected]
    links = (graph | graph.T)[selected[:, 0].cpu(), selected[:, 1].cpu()].float().to(adapted.device)
    columns = torch.column_stack((scores[:, indices].T, affinity.min(-1).values, affinity.mean(-1), links))
    selected = selected.tolist()
    needed = None if sentence_cache is None else {i for pair in selected for i in pair}
    texts = sentence_bank(variant["context"], selected=needed, cached=sentence_cache, counts=sentence_counts)
    return mode, texts, query_terms, selected, columns.tolist(), winners


def requested_sentences(plans):
    sentences = {}
    for _id, _split, _gold, _q, variants in plans:
        for _mode, bank, _question, selected, _columns, _winners in variants:
            for idx in {i for pair in selected for i in pair}:
                for sentence in bank[idx][1]:
                    sentences.setdefault(sentence, len(sentences))
    return sentences


def flush(plans, enc, *, share_document_terms=False):
    sentences = requested_sentences(plans)
    if sentences:
        vectors = enc([" ".join(s) for s in sentences], "passage", max_length=96)
        # One transfer per chunk for all sentence–question affinities.
        affinities = (torch.stack([plan[3] for plan in plans]) @ vectors.T).tolist()
    else:
        affinities = [[] for _ in plans]
    result = []
    for (id_, split, gold, _q, variants), similarities in zip(plans, affinities, strict=True):
        output = {}
        for mode, bank, question, selected, columns, winners in variants:
            scores = {d: [similarities[sentences[s]] for s in bank[d][1]]
                      for d in {i for pair in selected for i in pair}}
            cache = {i: document_terms(question, bank, scores, i) for i in scores} if share_document_terms else None
            features = [base + sentence_columns(question, bank, scores, pair, document_cache=cache)
                        for base, pair in zip(columns, selected, strict=True)]
            gold_index = next((i for i, pair in enumerate(selected) if tuple(pair) == gold), -1)
            output[mode] = {"gold": gold_index, "winners": winners,
                            "features": [[round(x, 6) for x in r] for r in features]}
        result.append({"id": id_, "split": split, "variants": output})
    return result


def profile_plans(plans, enc, *, host_cache=False):
    """Time the added sentence stage for one request at a time, after shortlisting."""
    clean = [(*plan[:4], [plan[4][0]]) for plan in plans]
    assert all(plan[4][0][0] == "clean" for plan in clean)
    corpus = [" ".join(tokens) for tokens in requested_sentences(clean)]
    cache = enc(corpus, "passage", max_length=96).contiguous()
    positions = {sentence: i for i, sentence in enumerate(corpus)}
    barrier = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(cache.device.type)

    def cached(texts, _prefix, max_length):
        assert max_length == 96
        index = torch.tensor([positions[text] for text in texts], device=cache.device)
        return cache.index_select(0, index)

    modes = [("cold", enc), ("resident", cached)]
    if host_cache:
        host_bank = cache.cpu()
        limit = max(len(requested_sentences([plan])) for plan in clean)
        staging = torch.empty((limit, cache.shape[1]), dtype=cache.dtype, device="cpu", pin_memory=True)

        def host_cached(texts, _prefix, max_length):
            assert max_length == 96
            index = torch.tensor([positions[text] for text in texts], device="cpu")
            chunk = staging[:len(index)]
            torch.index_select(host_bank, 0, index, out=chunk)
            return chunk.to(cache.device, non_blocking=True)

        modes.append(("host_staged", host_cached))

    def narrow(plan, depth):
        mode, bank, question, selected, features, winners = plan[4][0]
        keep = set(winners)
        for head in range(2):
            keep.update(sorted(range(len(features)), key=lambda i: (-features[i][head], i))[:depth])
        keep = sorted(keep)
        return (*plan[:4], [(mode, bank, question, [selected[i] for i in keep], [features[i] for i in keep],
                             [keep.index(winner) for winner in winners])])

    for _, mode in modes:
        for depth in (4, 8, 16):
            flush([narrow(clean[0], depth)], mode)
    output = []
    for i, plan in enumerate(clean):
        for depth in (4, 8, 16):
            batch = [narrow(plan, depth)]
            row = {"id": plan[0], "pairs": len(batch[0][4][0][3]), "depth": depth,
                   "unique_sentences": len(requested_sentences(batch))}
            reference = {}
            for label, mode in (modes if i % 2 == 0 else modes[::-1]):
                if barrier:
                    barrier()
                t0 = perf_counter()
                reference[label] = flush(batch, mode)[0]["variants"]["clean"]
                if barrier:
                    barrier()
                row[label + "_ms"] = (perf_counter() - t0) * 1000
            x = reference["cold"]["features"]
            row["feature_max_abs_delta"] = max(abs(a - b) for label, _ in modes[1:]
                                               for u, v in zip(x, reference[label]["features"], strict=True)
                                               for a, b in zip(u, v, strict=True))
            output.append(row)
    return {"rows": output, "resident_cache_sentences": len(corpus), "host_cache": host_cache,
            "vector_bytes": cache.element_size() * cache.shape[1]}


def reader_requests(plans, features, contexts, params, reader, prompts):
    mean, scale, coefficients = (np.asarray(params[k], dtype=np.float64) for k in ("mean", "scale", "coefficients"))
    output = []
    for plan, row in zip(plans, features, strict=True):
        id_ = row["id"]
        selections = {v[0]: v[3] for v in plan[4]}
        answer, variants = contexts.pop(id_)
        responses = {}
        for mode, item in row["variants"].items():
            data = np.asarray(item["features"], dtype=np.float64)
            score = ((data - mean) / scale) @ coefficients
            shortlist = np.argsort(-data[:, :2], axis=0, kind="stable")[:4]
            reduced = np.unique(np.concatenate((shortlist.ravel(), item["winners"])))
            choices = {"full": item["winners"][0], "sentence16": int(score.argmax()),
                       "sentence4": int(reduced[score[reduced].argmax()])}
            case, affinity = variants[mode]
            responses[mode] = {}
            for label, index in choices.items():
                pair = sorted(selections[mode][index], key=lambda j: (-affinity[j], j))
                record = {"evidence": index == item["gold"]}
                prompts.setdefault(reader.prompt(case, pair), []).append((record, answer))
                responses[mode][label] = record
        output.append({"id": id_, "split": row["split"], "variants": responses})
    return output


@torch.inference_mode()
def evaluate(a, fixture, rows):
    enc = Encoder(a.assets / "encoder", 128, a.device)
    cfg = read(a.config)
    backend = {} if a.device == "cuda" else {"device": a.device}
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None,
                                    weights_sha256=a.baseline_sha256, **backend)
    fitted = torch.load(a.heads, map_location="cpu", weights_only=True)
    ablations = torch.load(a.ablations, map_location="cpu", weights_only=True)
    assert fitted["provenance"]["models"] == read(a.assets / "revisions.json")
    assert ablations["reference_heads_sha256"] == digest(a.heads)
    full = ContextScorer().to(a.device).eval()
    full.load_state_dict(fitted["heads"]["context_only"])
    without = ContextScorer().to(a.device).eval()
    control = ablations["arms"]["refit_no_coverage"]
    without.load_state_dict(control["weights"])
    heads = full, without, torch.tensor(control["mask"], device=a.device)
    donors = [(d["title"], [d["text"]]) for d in fixture["donors"]]
    donor_words = [f" {' '.join(words(doc['text']))} " for doc in fixture["donors"]]
    donor_keys = enc(candidate_texts({"context": donors}), "passage")
    queries = enc([r["question"] for r, _ in rows], "query")
    supplied = enc([text for r, _ in rows for text in candidate_texts(r)], "passage").reshape(len(rows), 10, -1)
    pairs = torch.triu_indices(64, 64, 1).to(a.device)
    result, pending = [], []
    reader, contexts, prompts = None, {}, {}
    if a.reader_policy:
        reader = Reader(a.assets / "reader", a.device)

    def finish_batch():
        features = flush(pending, enc)
        if reader is not None:
            features = reader_requests(pending, features, contexts, a.policy, reader, prompts)
        result.extend(features)
        pending.clear()

    for (row, split), query, keys in zip(rows, queries.split(1), supplied, strict=True):
        context = None
        if reader is not None:
            context = {}
            contexts[row["id"]] = (row["answer"], context)
        pending.append(plan_case(row, split, query, keys, donor_keys, donors, donor_words, adapter, enc, heads, pairs,
                                 contextual_test=a.contextual_test, reader_context=context))
        if len(pending) == 16 and not a.profile:
            finish_batch()
    if a.profile:
        return profile_plans(pending, enc, host_cache=a.profile_host)
    if pending:
        finish_batch()
    if reader is not None:
        for predicted, entries in zip(reader.answers(list(prompts)), prompts.values(), strict=True):
            for record, truth in entries:
                record["exact"] = bool(measure_answer(predicted, truth)["em"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "config", "baseline", "heads", "ablations", "fixture", "splits", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--dataset", choices=("twowiki", "musique"), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--contextual-test", action="store_true", help="Evaluate held-out context-bearing false references")
    mode.add_argument("--profile", action="store_true", help="Time the single-request sentence stage on a held-out sample")
    mode.add_argument("--profile-host", action="store_true", help="Add a host cache with a pinned transfer buffer")
    mode.add_argument("--reader-policy", type=Path, help="Evaluate fixed evidence selection and the frozen answer reader")
    p.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    if a.profile_host and a.device != "cuda":
        p.error("The pinned host transfer profile requires CUDA")
    a.profile = a.profile or a.profile_host
    fixture, splits = read(a.fixture), read(a.splits)
    if a.reader_policy:
        a.policy = read(a.reader_policy)
        assert a.policy["fixture_sha256"] == digest(a.fixture) and a.policy["split_sha256"] == digest(a.splits)
        assert a.policy["features"] == list(COLUMNS)
    rows = [(r, splits[r["id"]]) for r in fixture["rows"][a.dataset]
            if splits[r["id"]].endswith("-test") or
            (a.dataset == "twowiki" and not (a.contextual_test or a.profile or a.reader_policy))]
    if a.profile:
        rows = sorted(rows, key=lambda entry: stable("kam-verifier-profile-v1:" + entry[0]["id"]))[:48]
    rows = rows[a.shard::a.shards]
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    result = evaluate(a, fixture, rows)
    hardware = torch.cuda.get_device_name() if a.device == "cuda" else a.device
    if a.reader_policy:
        save(a.output, {"dataset": a.dataset, "device": hardware,
                        "fixture_sha256": digest(a.fixture), "policy_sha256": digest(a.reader_policy),
                        "seconds": perf_counter() - start, "rows": result})
        return
    if a.profile:
        save(a.output, {"dataset": a.dataset, "torch": str(torch.__version__), "device": hardware,
                        "fixture_sha256": digest(a.fixture), **result})
        return
    save(a.output, {"dataset": a.dataset, "columns": COLUMNS, "shortlist_per_head": SHORTLIST,
                    "contextual_test": a.contextual_test,
                    "counts": Counter(r["split"] for r in result), "fixture_sha256": digest(a.fixture),
                    "split_sha256": digest(a.splits), "heads_sha256": digest(a.heads),
                    "ablations_sha256": digest(a.ablations), "baseline_sha256": a.baseline_sha256,
                    "device": hardware, "seconds": perf_counter() - start, "rows": result})


if __name__ == "__main__":
    main()
