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
GUARD_COLUMNS = ("shared_document", "added_document_count", "added_document_rank", "union_has_anchor",
                 "broad_margin", "union_margin", "union_original_pair_margin", "original_pair_in_shortlist",
                 "broad_pair_link", "union_pair_link", "weak_semantic_shift", "mean_affinity_shift",
                 "mention_margin_shift", "union_size")


def guard_vector(arms, winners, first, original_size, hits, union_size):
    """Return decision-time information before either answer has been generated."""
    broad_pair, union_pair = (set(arms[key]["pair"]) for key in ("broad", "conditioned_union"))
    _, _, broad_scores, broad_x = winners["broad"]
    slots, pairs, union_scores, union_x = winners["conditioned_union"]
    additions = {doc for doc, slot in zip(arms["conditioned_union"]["pair"], slots, strict=True)
                 if slot >= original_size}
    count = len(additions)
    original_only = [v for pair, v in zip(pairs, union_scores, strict=True) if max(pair) < original_size]
    gap = max(union_scores) - max(original_only) if original_only else 0.
    margins = [max(x) - np.partition(x, -2)[-2] for x in (broad_scores, union_scores)]
    return [round(float(value), 7) for value in (
        len(broad_pair & union_pair), count, min((hits.index(d) + 1 for d in additions), default=0) / 64,
        first in union_pair, *margins, gap, bool(original_only), broad_x[4], union_x[4],
        union_x[11] - broad_x[11], union_x[3] - broad_x[3], union_x[15] - broad_x[15], union_size,
    )]


def union_example(index, example, original, top):
    original_ids = set(original)
    extra = [doc for doc in top if doc not in original_ids]
    extension = index.candidates(example, extra)["context"] if extra else []
    context = example["context"] + [(title, SENTENCE_BREAK.split(body)) for title, (body,) in extension]
    return original + extra, {"question": example["question"], "context": context}


def shared_relations(variants, pair_indices, original_size, retained_pairs, *, vectorized=False):
    from kam_union_relation_features import bitset_relation_features, induced_original_features

    ranked, variant = variants["conditioned_union"]
    features = bitset_relation_features(variant, pair_indices[len(ranked)][0], vectorized=vectorized)
    return {"broad": induced_original_features(features, original_size, retained_pairs),
            "conditioned_union": features}


def profile_relations(variants, pair_indices, original_size, retained_pairs, reference_first):
    """Compare relation construction under identical text and candidate inputs."""
    def reference():
        return {arm: relation_features(variant, pair_indices[len(ranked)][0])
                for arm, (ranked, variant) in variants.items()}

    def shared():
        return shared_relations(variants, pair_indices, original_size, retained_pairs)

    methods = {"reference": reference, "shared": shared}
    order = tuple(methods) if reference_first else tuple(reversed(methods))
    features, elapsed = {}, {}
    for name in order:
        started = perf_counter()
        features[name] = methods[name]()
        elapsed[name] = (perf_counter() - started) * 1000
    for arm in variants:
        assert all(torch.equal(a, b) for a, b in zip(features["reference"][arm], features["shared"][arm], strict=True))
    return features["shared"], elapsed


@torch.inference_mode()
def run(a):
    started = perf_counter()
    cfg, fixture, manifest = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json"))
    sealed = load_gzip(a.pools)
    index_sha = digest(a.inputs / "lexical" / "completed.json")
    assert sealed["fixture_sha256"] == digest(a.fixture) and sealed["index_sha256"] == index_sha
    assert fixture["source_sha256"] == manifest["source_sha256"]
    split = a.guard_split or "test"
    cases = fixture["rows"][split]
    assert set(sealed["rows"]) == {q["id"] for group in fixture["rows"].values() for q in group}
    assert len(cases) == (3072 if split == "training" else 1024)
    precomputed = {}
    if a.precomputed_corpus_hits:
        moved = load_gzip(a.precomputed_corpus_hits)
        assert moved["fixture_sha256"] == sealed["fixture_sha256"] and moved["index_sha256"] == index_sha
        assert moved["pools_sha256"] == digest(a.pools) and moved["fitted_sha256"] == digest(a.fitted)
        precomputed = moved["rows"]
        assert set(precomputed) == {q["id"] for q in cases}
    anchors = {}
    if a.anchor_reference:
        reference = load_gzip(a.anchor_reference)
        assert all(s["fixture_sha256"] == sealed["fixture_sha256"] and
                   s["pools_sha256"] == digest(a.pools) and s["fitted_sha256"] == digest(a.fitted)
                   for s in reference)
        anchors = {r["id"]: r["arms"]["broad"]["pair"][0] for s in reference for r in s["rows"]}
        assert set(anchors) == {q["id"] for q in cases}
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
    answer_readers = {"fp32": reader}
    if a.compare_reader_precision:
        bf16 = Reader(a.assets / "reader")
        bf16.model.bfloat16()
        answer_readers["bf16"] = bf16
    sentence_encoders = {"fp32": encoder}
    detailed_profile = (a.compare_prompt_cache or a.compare_feature_reuse or a.compare_second_batch
                        or a.compare_int8_initial)
    sustained_batch = (a.confirm_fp16_throughput or a.compare_batched_initial or a.compare_reader_precision
                       or detailed_profile or a.sweep_optimized_widths)
    low_precision = {"bf16": torch.bfloat16, "fp16": torch.float16} if a.compare_sentence_precision else (
        {"fp16": torch.float16} if sustained_batch else {})
    for label, dtype in low_precision.items():
        resident = Encoder(a.assets / "encoder", 128)
        resident.model.to(dtype=dtype)
        sentence_encoders[label] = resident
    index = FullWikiIndex(a.inputs, manifest, cfg)
    corpus_engine, int8_bytes = None, None
    if a.corpus_profile_index:
        from run_kam_fullwiki_coarse import Search

        partition = torch.load(a.corpus_profile_index, weights_only=True, map_location="cpu")
        assert partition["index_sha256"] == index_sha and partition["source_sha256"] == manifest["source_sha256"]
        corpus_engine = Search(index, partition)
        if a.compare_int8_initial:
            int8_bytes = corpus_engine.initialize_int8()
    pair_indices, original_rows = {}, {}
    if a.live_anchor_profile or a.live_initial_profile:
        from kam_union_relation_features import bitset_relation_features

    def pairs_for(n):
        if n not in pair_indices:
            pairs = torch.triu_indices(n, n, 1)
            pair_indices[n] = pairs, pairs.cuda()
        return pair_indices[n]

    def plan_variant(arm, ranked, variant, vector, terms, contexts, features=None, *,
                     sentence_cache=None, sentence_counts=None):
        cpu_pairs, gpu_pairs = pairs_for(len(ranked))
        graph, lexical = features if features is not None else relation_features(variant, cpu_pairs)
        bank = index.bank[torch.tensor(ranked, device="cuda")].unsqueeze(0)
        return rank_variant(arm, variant, graph, lexical, bank, vector, terms, heads, gpu_pairs, contexts,
                            sentence_cache=sentence_cache, sentence_counts=sentence_counts)

    def select(stage, item, contexts):
        arm, _, _, pairs, _, _ = stage
        x = np.asarray(item["variants"][arm]["features"], dtype=np.float64)
        scores = ((x - model["mean"]) / model["scale"]) @ model["coefficients"]
        order = contexts[arm][1]
        winner = int(scores.argmax())
        chosen = sorted(pairs[winner], key=lambda j: (-order[j], j))
        return chosen, pairs, scores, x[winner]

    def score_cases(chunk, offset, prompts, output, use_shared=False, *, live_anchor=False, phase=None,
                    reuse_sentences=False, sentence_counts=None, initial_method=None, corpus_method=None,
                    sparse_sentence_bank=False, normalization=None, bucket_sentences=False, sentence_precision="fp32",
                    cache_reader_passages=False, reuse_features=False):
        encode_sentences = encoder
        if sentence_counts is not None:
            cache = {}

            def encode_sentences(texts, prefix, max_length):
                sentence_counts["requested"] += len(texts)
                if not reuse_sentences:
                    sentence_counts["encoded"] += len(texts)
                    return encoder(texts, prefix, max_length=max_length)
                missing = [text for text in texts if text not in cache]
                sentence_counts["encoded"] += len(missing)
                if missing:
                    if bucket_sentences:
                        missing.sort(key=len)
                    if a.compare_sentence_bucketing or a.compare_sentence_precision:
                        started_encoding = perf_counter()
                    vectors = sentence_encoders[sentence_precision](missing, prefix, max_length=max_length).float()
                    if a.compare_sentence_bucketing or a.compare_sentence_precision:
                        torch.cuda.synchronize()
                        phase["sentence_encoding_ms"] = (perf_counter() - started_encoding) * 1000
                    cache.update(zip(missing, vectors, strict=True))
                return torch.stack([cache[text] for text in texts])

        detailed = detailed_profile
        if detailed:
            phase_start = perf_counter()
        if initial_method is not None:
            query = encoder([case["question"] for case in chunk], "query")
            if detailed:
                torch.cuda.synchronize()
                phase["initial_encode_ms"] = (perf_counter() - phase_start) * 1000
            search_start = perf_counter()
            depth = 128 if initial_method == "dense" else 64
            individual = query.split(1)
            if initial_method == "batched_rescore":
                original = corpus_engine.full_rescore_batch(query).tolist()
            elif initial_method == "batched_int8":
                original = corpus_engine.full_rescore_int8(query).tolist()
            elif initial_method == "dense_rescore":
                original = [corpus_engine.full_rescore(q, exclude=None)[0].tolist() for q in individual]
            else:
                original = [index.search(initial_method, c["question"], q, depth)[:64]
                            for c, q in zip(chunk, individual, strict=True)]
            if phase is not None:
                phase["initial_search_ms"] = (perf_counter() - search_start) * 1000
            examples = [pool_text(index.db, c["question"], ids) for c, ids in zip(chunk, original, strict=True)]
            if detailed:
                phase["initial_articles_ms"] = (perf_counter() - search_start) * 1000 - phase["initial_search_ms"]
        else:
            original = [sealed["rows"][c["id"]] for c in chunk]
            examples = [pool_text(index.db, q["question"], ids) for q, ids in zip(chunk, original, strict=True)]
            query = encoder([c["question"] for c in chunk], "query")
        adapted = adapter(query)
        if phase is not None:
            original_ready = perf_counter()
        early = None
        if live_anchor:
            case, = chunk
            base, = original
            example, = examples
            early_context = {}
            terms = set(words(case["question"])) - STOP
            relation = bitset_relation_features(example, pairs_for(len(base))[0])
            stage = plan_variant("broad", base, example, adapted, terms, early_context, relation)
            item, = flush([(case["id"], split, (), query[0], [stage])], encode_sentences)
            early = select(stage, item, early_context)
            first_slots = early[0][:1]
            first_ids = [base[first_slots[0]]]
        elif anchors:
            first_ids = [anchors[q["id"]] for q in chunk]
            first_slots = [pool.index(anchor) for pool, anchor in zip(original, first_ids, strict=True)]
        else:
            ids = torch.tensor(original, device="cuda")
            first_slots = score(adapted, index.bank[ids]).argmax(dim=-1, keepdim=True)
            first_ids = ids.gather(1, first_slots).flatten().tolist()
            first_slots = first_slots.flatten().tolist()
        if phase is not None:
            anchor_ready = perf_counter()
        if precomputed:
            selections = [precomputed[q["id"]] for q in chunk]
            assert first_ids == [row["first"] for row in selections]
            candidates = [row[key] for key in EXPANDED for row in selections]
            assert all(len(c) == len(set(c)) == 64 for c in candidates)
        else:
            augmented = [f"{c['question']} First retrieved document: {candidate_texts(c)[slot]}"
                         for c, slot in zip(examples, first_slots, strict=True)]
            second = adapter(encoder(augmented, "query"))
            conditioned = (adapted + second) * .5
            queries = conditioned if a.guard_split else torch.cat((adapted, conditioned))
            if corpus_method == "batched_fused_256":
                assert a.guard_split
                candidates = corpus_engine.fused_probe_batch(queries, first_ids, 256).tolist()
            elif corpus_method:
                if corpus_method == "exact":
                    search = corpus_engine.exact
                elif corpus_method == "full_rescore":
                    search = corpus_engine.full_rescore
                else:
                    implementation, count = corpus_method.split("_")
                    fn = corpus_engine.fused_probe if implementation == "fused" else corpus_engine.probe
                    search = lambda q, first: fn(q, first, int(count))
                candidates = [search(q, first)[0].tolist() for q, first in zip(queries.split(1), first_ids, strict=True)]
            else:
                candidates, _ = top_corpus(index, queries, first_ids if a.guard_split else first_ids * 2)

        if phase is not None:
            search_ready = perf_counter()
        plans, details = [], []
        for i, (case, example, base) in enumerate(zip(chunk, examples, original, strict=True)):
            gold = set(case["gold_ids"])
            assert ({base[j] for j, (title, _) in enumerate(example["context"])
                     if title_key(title) in case["gold_titles"]} == gold.intersection(base))
            id_ = case["id"]
            hits = ({"conditioned_union": candidates[i]} if a.guard_split else
                    dict(zip(EXPANDED, (candidates[i], candidates[i + len(chunk)]), strict=True)))
            variants = {"broad": (base, example)} | {
                arm: union_example(index, example, base, top) for arm, top in hits.items()
            }
            relations, elapsed = {}, {}
            if (a.verify_shared_relation or use_shared) and not live_anchor:
                expanded_size = len(variants["conditioned_union"][0])
                if expanded_size not in original_rows:
                    expanded_pairs = pairs_for(expanded_size)[0]
                    keep = expanded_pairs[1] < len(base)
                    assert torch.equal(expanded_pairs[:, keep], pairs_for(len(base))[0])
                    original_rows[expanded_size] = keep
                if a.verify_shared_relation:
                    relations, elapsed = profile_relations(variants, pair_indices, len(base), original_rows[expanded_size],
                                                           (offset + i) % 2 == 0)
                else:
                    relations = shared_relations(variants, pair_indices, len(base), original_rows[expanded_size],
                                                 vectorized=reuse_features)
            prepared, contexts = [], {}
            sentence_cache = {} if sparse_sentence_bank else None
            terms = set(words(case["question"])) - STOP
            for arm, (ranked, variant) in variants.items():
                if live_anchor and arm == "broad":
                    continue
                features = relations.get(arm)
                if live_anchor:
                    features = bitset_relation_features(variant, pairs_for(len(ranked))[0])
                prepared.append(plan_variant(arm, ranked, variant, adapted[i:i+1], terms, contexts, features,
                                             sentence_cache=sentence_cache, sentence_counts=normalization))
            plans.append((id_, split, (), query[i], prepared))
            details.append((case, base, variants, contexts, hits, first_ids[i], elapsed))

        if detailed:
            torch.cuda.synchronize()
            prepared_at = perf_counter()
            phase["rerank_prepare_ms"] = (prepared_at - search_ready) * 1000
        scored = flush(plans, encode_sentences, share_document_terms=reuse_features)
        if detailed:
            torch.cuda.synchronize()
            phase["rerank_score_ms"] = (perf_counter() - prepared_at) * 1000
            if a.compare_feature_reuse:
                phase["rerank_features_sha256"] = sha256(json.dumps(scored, separators=(",", ":")).encode()).hexdigest()
        passage_cache = {} if cache_reader_passages else None
        passage_requests = prompt_seconds = 0
        if a.compare_sentence_bucketing or a.compare_sentence_precision:
            phase["rerank_through_sentence_ms"] = (perf_counter() - search_ready) * 1000
        for item, plan, detail in zip(scored, plans, details, strict=True):
            case, base, variants, contexts, hits, first, elapsed = detail
            gold = set(case["gold_ids"])
            arms = {}
            winners = {stage[0]: select(stage, item, contexts) for stage in plan[4]}
            if early is not None:
                winners = {"broad": early, **winners}
            for arm, (chosen, pairs, _, _) in winners.items():
                ranked, example = variants[arm]
                shortlisted = any({ranked[j], ranked[k]} == gold for j, k in pairs)
                arms[arm] = {"pair": [ranked[j] for j in chosen], "gold_pair_in_shortlist": shortlisted}
                if detailed:
                    before_prompt = perf_counter()
                    passage_requests += len(chosen)
                prompt = reader.prompt(example, chosen, passage_cache=passage_cache)
                if detailed:
                    prompt_seconds += perf_counter() - before_prompt
                prompts.setdefault(prompt, []).append((arms[arm], case["answer"]))
            row = {"id": case["id"], "first": first, "corpus_hits": hits, "arms": arms,
                   "pool_hash": sha256(json.dumps(base, separators=(",", ":")).encode()).hexdigest()}
            if initial_method is not None:
                row["initial_ids"] = base
            if a.guard_split:
                row["features"] = guard_vector(arms, winners, first, len(base), hits["conditioned_union"],
                                               len(variants["conditioned_union"][0]))
            if a.verify_shared_relation:
                row["relation_timing_ms"] = elapsed
            output.append(row)
        if detailed:
            phase.update(prompt_build_ms=prompt_seconds * 1000, passage_requests=passage_requests,
                         passage_tokenized=passage_requests if passage_cache is None else len(passage_cache),
                         prompts_sha256=sha256(json.dumps(list(prompts), ensure_ascii=False).encode()).hexdigest())
        if phase is not None:
            phase.update(anchor_ms=(anchor_ready - original_ready) * 1000,
                         conditioned_search_ms=(search_ready - anchor_ready) * 1000,
                         remaining_rerank_ms=(perf_counter() - search_ready) * 1000)

    def read_answers(prompts, precision="fp32"):
        answers = answer_readers[precision].answers(list(prompts))
        for predicted, requests in zip(answers, prompts.values(), strict=True):
            for arm, truth in requests:
                arm["exact"] = bool(measure_answer(predicted, truth)["em"])

    def throughput_block(cases, width, precision="fp32", initial_method="dense_rescore", reader_precision="fp32",
                         cache_reader_passages=False, reuse_features=False, batch_second=False):
        rows, timing = [], []
        counts = dict(requested=0, encoded=0)
        if a.sweep_optimized_widths:
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        for offset in range(0, len(cases), width):
            prompts, block_rows = {}, []
            sentence_counts = dict(requested=0, encoded=0)
            phases = {} if (a.compare_batched_initial or a.compare_reader_precision or
                            detailed_profile or a.sweep_optimized_widths) else None
            torch.cuda.synchronize()
            begin = perf_counter()
            score_cases(cases[offset:offset+width], offset, prompts, block_rows, True, reuse_sentences=True,
                        sentence_counts=sentence_counts, initial_method=initial_method,
                        corpus_method="batched_fused_256" if batch_second else "fused_256",
                        sparse_sentence_bank=True, sentence_precision=precision, phase=phases,
                        cache_reader_passages=cache_reader_passages, reuse_features=reuse_features)
            rank_done = perf_counter()
            read_answers(prompts, reader_precision)
            torch.cuda.synchronize()
            done = perf_counter()
            timing.append({"size": len(block_rows), "request_ms": (done - begin) * 1000,
                           "reader_ms": (done - rank_done) * 1000})
            if phases is not None:
                timing[-1]["initial_search_ms"] = phases["initial_search_ms"]
            if detailed_profile:
                timing[-1].update(phases)
            rows.extend(block_rows)
            for key, value in sentence_counts.items():
                counts[key] += value
        output = {"rows": rows, "timing": timing, "sentence_counts": counts}
        if a.sweep_optimized_widths:
            output["extra_peak_bytes"] = torch.cuda.max_memory_allocated() - resident
        return output

    output = []
    def request(case, i, use_shared, *, live_anchor=False, phase=None, reuse_sentences=False,
                initial_method=None, corpus_method=None, sparse_sentence_bank=False, bucket_sentences=False,
                sentence_precision="fp32"):
        prompts, rows = {}, []
        sentence_counts = dict(requested=0, encoded=0) if a.sentence_reuse_profile or reuse_sentences else None
        normalization = dict(candidate_documents=0, normalized_documents=0) if a.compare_shortlist_text else None
        torch.cuda.synchronize()
        before = perf_counter()
        score_cases([case], i, prompts, rows, use_shared=use_shared, live_anchor=live_anchor, phase=phase,
                    reuse_sentences=reuse_sentences, sentence_counts=sentence_counts, initial_method=initial_method,
                    corpus_method=corpus_method, sparse_sentence_bank=sparse_sentence_bank, normalization=normalization,
                    bucket_sentences=bucket_sentences, sentence_precision=sentence_precision)
        before_reader = perf_counter()
        read_answers(prompts)
        torch.cuda.synchronize()
        done = perf_counter()
        if phase is not None:
            phase["reader_ms"] = (done - before_reader) * 1000
        if sentence_counts is not None:
            rows[0]["sentence_counts"] = sentence_counts
        if normalization is not None:
            rows[0]["sentence_normalization"] = normalization
        return rows[0], (done - before) * 1000

    index_profile = {}
    if corpus_engine is not None:
        import gc
        from random import Random

        from benchmark_kam_article_prewarm import read_index

        bytes_scanned, readiness_seconds = read_index(a.inputs / "lexical" / "articles.sqlite")
        methods = {mode: ("dense", mode) for mode in ("exact", "full_rescore", "probe_64", "probe_256")}
        if a.corpus_fused:
            methods.update({mode: ("dense", mode) for mode in ("fused_64", "fused_256")})
        if a.compare_rescored_initial:
            methods = {f"{first}_{second}": (initial, second)
                       for first, initial in (("exact_initial", "dense"), ("rescore_initial", "dense_rescore"))
                       for second in ("full_rescore", "fused_256")}
        if a.compare_shortlist_text:
            methods = dict(all_text=("dense_rescore", "fused_256"), shortlist_text=("dense_rescore", "fused_256"))
        if a.compare_sentence_bucketing:
            methods = dict(unbucketed=("dense_rescore", "fused_256"), bucketed=("dense_rescore", "fused_256"))
        if a.compare_sentence_precision:
            methods = {name: ("dense_rescore", "fused_256") for name in sentence_encoders}

        def profile(case, i, method, phase):
            initial, second = methods[method]
            return request(case, i, True, phase=phase, initial_method=initial, corpus_method=second,
                           reuse_sentences=True, sparse_sentence_bank=(method == "shortlist_text" or
                                                                      a.compare_sentence_bucketing or a.compare_sentence_precision),
                           bucket_sentences=method == "bucketed",
                           sentence_precision=method if a.compare_sentence_precision else "fp32")

        rng = Random(195823 + a.shard)
        if a.compare_request_throughput or sustained_batch:
            sample = rng.sample(cases[4:], 48)
            trial = {str(w): (w, "fp32") for w in (1, 2, 4, 8)}
            if sustained_batch:
                trial = {f"{p}/{w}": (w, p) for p in ("fp32", "fp16") for w in (1, 8)}
                if a.compare_batched_initial:
                    trial = {"separate": (8, "fp16"), "shared": (8, "fp16", "batched_rescore")}
                if a.compare_reader_precision:
                    trial = {p: (8, "fp16", "batched_rescore", p) for p in answer_readers}
                if a.compare_prompt_cache:
                    trial = {"baseline": (8, "fp16", "batched_rescore", "fp32", False),
                             "cached_prompt": (8, "fp16", "batched_rescore", "fp32", True)}
                if a.compare_feature_reuse:
                    trial = {"baseline": (8, "fp16", "batched_rescore", "fp32", False, False),
                             "shared_features": (8, "fp16", "batched_rescore", "fp32", False, True)}
                if a.compare_second_batch:
                    trial = {"baseline": (8, "fp16", "batched_rescore", "fp32", False, False, False),
                             "shared_second": (8, "fp16", "batched_rescore", "fp32", False, False, True)}
                if a.compare_int8_initial:
                    trial = {"baseline": (8, "fp16", "batched_rescore", "fp32", True, True),
                             "int8_initial": (8, "fp16", "batched_int8", "fp32", True, True)}
                if a.sweep_optimized_widths:
                    trial = {f"optimized/{width}": (width, "fp16", "batched_rescore", "fp32", True, True)
                             for width in (1, 4, 8, 16, 32)}
                    trial["reference/8"] = (8, "fp16", "batched_rescore")
                prior = {q["id"] for q in cases[:4] + sample}
                eligible = [q for q in cases if q["id"] not in prior]
                sample = rng.sample(eligible, 192 if a.sweep_optimized_widths else 200)
            group = 32 if a.sweep_optimized_widths else 8
            avoid = {c["id"] for c in sample}
            warm = [c for c in cases if c["id"] not in avoid][:group]
            for config in trial.values():
                throughput_block(warm, *config)
            gc.collect(2)
            for i in range(0, len(sample), group):
                order = rng.sample(list(trial), len(trial))
                batch = sample[i:i+group]
                output.append({"ids": [r["id"] for r in batch], "order": order,
                               "batches": {key: throughput_block(batch, *trial[key]) for key in order}})
        else:
            for i, case in enumerate(cases[:4]):
                for method in methods:
                    profile(case, i, method, {})
            gc.collect(2)
            sample = rng.sample(cases[4:], 48)
            for i, case in enumerate(sample):
                order = rng.sample(list(methods), len(methods))
                outcomes, latencies, details = {}, {}, {}
                for method in order:
                    details[method] = {}
                    outcomes[method], latencies[method] = profile(case, i, method, details[method])
                for initial in {group[0] for group in methods.values()}:
                    group = [outcomes[p] for p in methods if methods[p][0] == initial]
                    assert all(q["initial_ids"] == group[0]["initial_ids"] and q["first"] == group[0]["first"] for q in group)
                output.append({"id": case["id"], "order": order, "request_ms": latencies, "phases_ms": details,
                               "records": outcomes})
        index_profile = {"corpus_profile_partition": digest(a.corpus_profile_index), "document_bytes": bytes_scanned,
                         "document_readiness_seconds": readiness_seconds}
        if a.compare_rescored_initial or a.compare_shortlist_text or a.compare_sentence_bucketing or a.compare_sentence_precision:
            index_profile["search_modes"] = methods
        if sustained_batch:
            index_profile["batch_policies"] = trial
        if int8_bytes is not None:
            index_profile["int8_first_pass_bytes"] = int8_bytes
    elif a.index_prewarm_rank is not None:
        from os import uname
        from random import Random

        from benchmark_kam_article_prewarm import barrier, read_index

        rank = a.index_prewarm_rank
        sample = Random(20261127).sample(sorted(cases, key=lambda c: c["id"]), 256)
        selected_ids = {q["id"] for q in sample}
        warmup = [q for q in cases if q["id"] not in selected_ids][:4]
        for i, q in enumerate(warmup):
            request(q, i, True)
        for i, q in enumerate(sample[rank::2]):
            row, latency = request(q, i, True)
            output.append(row | {"index_warmed": False, "request_ms": latency})
        barrier(a.index_prewarm_jobdir, rank)
        database_bytes, seconds = read_index(a.inputs / "lexical" / "articles.sqlite")
        warm_barrier = a.index_prewarm_jobdir / "warmed"
        warm_barrier.mkdir(exist_ok=True)
        barrier(warm_barrier, rank)
        for i, q in enumerate(sample[1 - rank::2]):
            row, latency = request(q, i, True)
            output.append(row | {"index_warmed": True, "request_ms": latency})
        index_profile = {"index_profile_rank": rank, "database_bytes": database_bytes, "prewarm_seconds": seconds,
                         "node_sha256": sha256(uname().nodename.encode()).hexdigest()}
    elif a.live_anchor_profile or a.live_initial_profile:
        from random import Random

        from benchmark_kam_article_prewarm import read_index

        database_bytes, seconds = read_index(a.inputs / "lexical" / "articles.sqlite")
        engines, lexical_profile = {}, {}
        if a.warm_lexical_profile:
            import tantivy

            lexical = a.inputs / "lexical" / "tantivy"
            started_lexical = perf_counter()
            files = sorted(p for p in lexical.iterdir() if p.is_file() and not p.name.startswith("."))
            lexical_bytes = sum(read_index(path)[0] for path in files)
            lexical_profile = {"lexical_prewarm_bytes": lexical_bytes,
                               "lexical_prewarm_seconds": perf_counter() - started_lexical}
            threaded = tantivy.Index.open(str(lexical), search_threads=2)
            engines = {"hybrid_serial": (index.index, index.searcher),
                       "hybrid_two_threads": (threaded, threaded.searcher())}
            assert all(s.num_docs == index.meta["indexed"] for _, s in engines.values())
            policies = ["dense", *engines]
        elif a.live_initial_profile:
            policies = ["dense", "hybrid"]
        elif a.sentence_reuse_profile:
            policies = ["live_reuse"] if a.gc_sustained else ["live_broad", "live_reuse"]
        else:
            policies = ["affinity", "live_broad"]

        def profile_request(case, i, arm, phase):
            if arm in engines:
                index.index, index.searcher = engines[arm]
            method = "hybrid" if arm in engines else arm
            return request(case, i, True, live_anchor=arm != "affinity", phase=phase,
                           reuse_sentences=a.live_initial_profile or arm == "live_reuse",
                           initial_method=method if a.live_initial_profile else None)

        for i, case in enumerate(cases[:4]):
            for arm in policies:
                profile_request(case, i, arm, {})
        collection_profile = {}
        if a.gc_readiness is not None:
            import gc
            from resource import RUSAGE_SELF, getrusage

            assert gc.isenabled()
            before_readiness = perf_counter()
            freed = gc.collect(2) if a.gc_readiness in ("collect", "freeze") else 0
            if a.gc_readiness == "freeze":
                gc.freeze()
            collection_profile = {"gc_readiness_mode": a.gc_readiness,
                                  "gc_readiness_seconds": perf_counter() - before_readiness,
                                  "gc_readiness_collected": freed, "gc_frozen_objects": gc.get_freeze_count(),
                                  "rss_before_requests_kib": getrusage(RUSAGE_SELF).ru_maxrss}
        rng = Random(23847 + a.shard)
        sample = cases[4:] if a.gc_sustained else cases[4:132:4]
        if a.gc_sustained:
            from os import uname

            def resident_kib():
                lines = Path("/proc/self/status").read_text().splitlines()
                return int(next(line.split()[1] for line in lines if line.startswith("VmRSS:")))

            assert len(sample) == 252
            collection_profile["sustained_gc"] = True
            collection_profile["node_sha256"] = sha256(uname().nodename.encode()).hexdigest()
            collection_profile["resident_kib_by_request"] = [[0, resident_kib()]]
        for i, case in enumerate(sample):
            arms = policies.copy()
            rng.shuffle(arms)
            timings, records, phases = {}, {}, {}
            for arm in arms:
                phases[arm] = {}
                if a.trace_gc:
                    import gc

                    events, gc_start = [], 0.0

                    def observe_gc(event, info):
                        nonlocal gc_start
                        if event == "start":
                            gc_start = perf_counter()
                        else:
                            events.append({"generation": info["generation"], "collected": info["collected"],
                                           "ms": (perf_counter() - gc_start) * 1000})

                    gc.callbacks.append(observe_gc)
                    try:
                        records[arm], timings[arm] = profile_request(case, i, arm, phases[arm])
                    finally:
                        gc.callbacks.remove(observe_gc)
                    if a.gc_sustained:
                        record = records.pop(arm)
                        compact = {
                            "exact_sha256": sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                            "decision_sha256": sha256(json.dumps({
                                "pool_hash": record["pool_hash"],
                                "first": record["first"],
                                "corpus_hits": {k: sorted(v) for k, v in record["corpus_hits"].items()},
                                "arms": record["arms"],
                            }, sort_keys=True).encode()).hexdigest(),
                            "gc_events": events,
                        }
                        records[arm] = compact
                        del record
                    else:
                        records[arm]["gc_events"] = events
                else:
                    records[arm], timings[arm] = profile_request(case, i, arm, phases[arm])
            output.append({"id": case["id"], "order": arms, "timing_ms": timings,
                           "phase_ms": phases, "records": records})
            if a.gc_sustained and ((i + 1) % 16 == 0 or i + 1 == len(sample)):
                collection_profile["resident_kib_by_request"].append([i + 1, resident_kib()])
        if collection_profile:
            collection_profile["rss_after_requests_kib"] = getrusage(RUSAGE_SELF).ru_maxrss
        index_profile = ({"prewarm_seconds": seconds, "database_bytes": database_bytes}
                         | lexical_profile | collection_profile)
    elif a.serving_profile:
        for i, case in enumerate(cases[:4]):
            request(case, i, False)
            request(case, i, True)
        for i, case in enumerate(cases[4:132:4]):
            order = (False, True) if i % 2 == 0 else (True, False)
            if a.serving_prime:
                request(case, i, order[0])
            measured = {shared: request(case, i, shared) for shared in order}
            baseline, old_ms = measured[False]
            candidate, new_ms = measured[True]
            assert baseline == candidate, case["id"]
            output.append(candidate | {"serving_timing_ms": {"reference": old_ms, "shared": new_ms},
                                       "reference_first": not order[0]})
    else:
        prompts = {}
        for offset in range(0, len(cases), BATCH):
            score_cases(cases[offset:offset+BATCH], offset, prompts, output)
        read_answers(prompts)
    return {"fixture_sha256": digest(a.fixture), "pools_sha256": digest(a.pools), "config_sha256": digest(a.config),
            "index_sha256": index_sha, "fitted_sha256": digest(a.fitted), "heads_sha256": digest(a.heads),
            "ablations_sha256": digest(a.ablations), "baseline_sha256": a.baseline_sha256, "models": revisions,
            "indexed_documents": index.meta["indexed"], "shard": a.shard, "shards": a.shards,
            "gpu": torch.cuda.get_device_name(), "seconds": perf_counter() - started, "rows": output} | (
                {"guard_columns": GUARD_COLUMNS, "split": split} if a.guard_split else {}) | (
                {"anchor_reference_sha256": digest(a.anchor_reference)} if a.anchor_reference else {}) | (
                {"serving_cache": "primed" if a.serving_prime else "first_pass"} if a.serving_profile else {}) | (
                {"corpus_hits_sha256": digest(a.precomputed_corpus_hits)} if precomputed else {}) | index_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "config", "fixture", "pools", "fitted", "baseline", "heads", "ablations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--guard-split", choices=("training", "test"))
    parser.add_argument("--verify-shared-relation", action="store_true")
    parser.add_argument("--anchor-reference", type=Path)
    parser.add_argument("--precomputed-corpus-hits", type=Path)
    parser.add_argument("--corpus-profile-index", type=Path)
    parser.add_argument("--corpus-fused", action="store_true")
    coarse_comparison = parser.add_mutually_exclusive_group()
    coarse_comparison.add_argument("--compare-rescored-initial", action="store_true")
    coarse_comparison.add_argument("--compare-shortlist-text", action="store_true")
    coarse_comparison.add_argument("--compare-sentence-bucketing", action="store_true")
    coarse_comparison.add_argument("--compare-sentence-precision", action="store_true")
    coarse_comparison.add_argument("--compare-request-throughput", action="store_true")
    coarse_comparison.add_argument("--confirm-fp16-throughput", action="store_true")
    coarse_comparison.add_argument("--compare-batched-initial", action="store_true")
    coarse_comparison.add_argument("--compare-reader-precision", action="store_true")
    coarse_comparison.add_argument("--compare-prompt-cache", action="store_true")
    coarse_comparison.add_argument("--compare-feature-reuse", action="store_true")
    coarse_comparison.add_argument("--compare-second-batch", action="store_true")
    coarse_comparison.add_argument("--sweep-optimized-widths", action="store_true")
    coarse_comparison.add_argument("--compare-int8-initial", action="store_true")
    parser.add_argument("--serving-profile", action="store_true")
    parser.add_argument("--serving-prime", action="store_true")
    live_profiles = parser.add_mutually_exclusive_group()
    live_profiles.add_argument("--live-anchor-profile", action="store_true")
    live_profiles.add_argument("--live-initial-profile", action="store_true")
    parser.add_argument("--sentence-reuse-profile", action="store_true")
    parser.add_argument("--warm-lexical-profile", action="store_true")
    parser.add_argument("--trace-gc", action="store_true")
    parser.add_argument("--gc-readiness", choices=("normal", "collect", "freeze"))
    parser.add_argument("--gc-sustained", action="store_true")
    parser.add_argument("--index-prewarm-rank", type=int, choices=(0, 1))
    parser.add_argument("--index-prewarm-jobdir", type=Path)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    a = parser.parse_args()
    assert 0 <= a.shard < a.shards
    if a.precomputed_corpus_hits and (
        a.guard_split or a.anchor_reference or a.serving_profile or a.index_prewarm_rank is not None
        or a.live_anchor_profile or a.live_initial_profile
    ):
        parser.error("precomputed corpus hits require the offline three-arm replay")
    if a.corpus_profile_index and (
        a.guard_split != "test" or a.shards != 4 or a.anchor_reference or a.serving_profile
        or a.index_prewarm_rank is not None or a.live_anchor_profile or a.live_initial_profile or a.verify_shared_relation
    ):
        parser.error("corpus timing requires the offline-configured test fixture and four live profiling shards")
    if a.corpus_fused and not a.corpus_profile_index:
        parser.error("fused corpus timing requires a coarse-partition checkpoint")
    if a.compare_rescored_initial and not a.corpus_profile_index:
        parser.error("rescored initial timing requires a coarse-partition checkpoint")
    if a.compare_shortlist_text and not a.corpus_profile_index:
        parser.error("sentence normalization timing needs separate live coarse-search mode")
    if a.compare_sentence_bucketing and not a.corpus_profile_index:
        parser.error("sentence bucketing timing requires a coarse-partition checkpoint")
    if a.compare_sentence_precision and not a.corpus_profile_index:
        parser.error("sentence precision timing requires a coarse-partition checkpoint")
    if not a.corpus_profile_index and (
        a.compare_request_throughput or a.confirm_fp16_throughput or a.compare_batched_initial
        or a.compare_reader_precision or a.compare_prompt_cache or a.compare_feature_reuse
        or a.compare_second_batch or a.sweep_optimized_widths or a.compare_int8_initial
    ):
        parser.error("batched request timing requires a coarse-partition checkpoint")
    if a.verify_shared_relation and a.guard_split != "test":
        parser.error("shared relation verification needs the two-arm test replay")
    if a.anchor_reference and a.guard_split != "test":
        parser.error("broad-ranked anchors need the two-arm test replay")
    if a.serving_profile and (a.guard_split != "test" or a.verify_shared_relation or a.anchor_reference):
        parser.error("serving timing needs the affinity-anchor two-arm test without parity double-computation")
    if a.serving_profile and a.shards != 4:
        parser.error("serving timing uses the fixed four-shard sampling design")
    if a.serving_prime and not a.serving_profile:
        parser.error("serving priming requires a serving profile")
    if a.index_prewarm_rank is not None and (
        a.guard_split != "test" or a.serving_profile or a.verify_shared_relation or a.anchor_reference
        or a.shards != 1 or not a.index_prewarm_jobdir
    ):
        parser.error("the shared-index crossover needs two independent full-cohort affinity workers")
    if a.warm_lexical_profile and not a.live_initial_profile:
        parser.error("warmed lexical timing requires the full initial-search profile")
    if a.gc_readiness is not None and not a.trace_gc:
        parser.error("collection readiness timing requires garbage collection tracing")
    if a.gc_sustained and not a.gc_readiness:
        parser.error("sustained collection timing requires a readiness mode")
    if a.trace_gc and not a.sentence_reuse_profile:
        parser.error("garbage collector tracing requires the sentence reuse profile")
    if a.sentence_reuse_profile and not a.live_anchor_profile:
        parser.error("sentence reuse timing requires the live anchor profile")
    if (a.live_anchor_profile or a.live_initial_profile) and (
        a.guard_split != "test" or a.anchor_reference or a.serving_profile or a.index_prewarm_rank is not None
        or a.verify_shared_relation or a.shards != 4
    ):
        parser.error("live timing uses the fixed four-shard warm-request design")
    torch.set_float32_matmul_precision("highest")
    if (a.compare_rescored_initial or a.compare_shortlist_text or a.compare_sentence_bucketing
            or a.compare_sentence_precision or a.compare_request_throughput or a.confirm_fp16_throughput
            or a.compare_batched_initial or a.compare_reader_precision or a.compare_prompt_cache or a.compare_feature_reuse or a.compare_second_batch or a.sweep_optimized_widths or a.compare_int8_initial):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    payload = json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode()
    a.output.write_bytes(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    main()
