#!/usr/bin/env python3
"""Compare evidence windows at fixed gold articles and reader context budgets."""

import argparse
import gzip
import json
from pathlib import Path
import re
from time import perf_counter

from audit_kam_natural_data import digest
from prepare_kam_news_windows import norm

ARMS = ("lead", "stride", "sentence_center", "two_centers", "gold_text", "no_evidence")
CAP = 192
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def pieces(body):
    words = body.split()
    return [" ".join(words[i:i+128]) for i in range(0, max(1, len(words) - 32), 96)]


def centered(n, place, width):
    start = max(0, min(place - width // 2, n - width))
    return start, min(n, start + width)


def windows(length, centers, ranks, budget, count):
    selected = [centers[ranks[0]]]
    if count == 2:
        outside = [centers[i] for i in ranks[1:] if abs(centers[i] - selected[0]) >= budget // 2]
        if outside:
            selected.append(outside[0])
    sizes = [budget // len(selected) + (i < budget % len(selected)) for i in range(len(selected))]
    merged = []
    for a, b in sorted(centered(length, c, size) for c, size in zip(selected, sizes, strict=True)):
        if merged and a <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(merged[-1][1], b)
        else:
            merged.append((a, b))
    assert sum(b - a for a, b in merged) <= budget
    return merged


def article_bank(corpus, ids, tokenizer, encoder):
    addresses, texts = {}, []
    for id_ in sorted(ids):
        article = corpus[id_]
        title = article["title"]
        title_ids = tokenizer(title + ".", add_special_tokens=False).input_ids[:64]
        pref = tokenizer.decode(title_ids)
        body = article["body"]
        strata = pieces(body)
        s0 = len(texts)
        texts.extend(f"{pref} {r}" for r in strata)
        raw = [p for p in SENTENCE.split(body) if p.strip()]
        tokenized = tokenizer(raw, add_special_tokens=False).input_ids
        offset = len(texts)
        spans, centers = [], []
        for row in tokenized:
            for j in range(0, len(row), 96):
                bit = row[j:j+96]
                centers.append(len(spans) + len(bit) // 2)
                spans.extend(bit)
                texts.append(f"{pref} {tokenizer.decode(bit)}")
        assert strata and centers and spans
        addresses[id_] = {"title_ids": title_ids, "strides": strata, "tokens": spans,
                          "centers": centers, "stride": (s0, offset), "sentences": (offset, len(texts))}
    return addresses, encoder(texts, "passage")


def select(a, vector, bank, tokenizer, annotations, article_limit, separator):
    tokens = a["tokens"]
    budget = article_limit - len(a["title_ids"]) - len(separator)
    assert budget > 0
    lo, hi = a["stride"]
    index = int((bank[lo:hi] @ vector).argmax())
    chosen = tokenizer(a["strides"][index], add_special_tokens=False).input_ids[:budget]
    low, high = a["sentences"]
    ranking = (bank[low:high] @ vector).argsort(descending=True, stable=True).tolist()
    result = {"lead": [tokens[:budget]], "stride": [chosen]}
    for key, count in (("sentence_center", 1), ("two_centers", 2)):
        result[key] = [tokens[i:j] for i, j in windows(len(tokens), a["centers"], ranking, budget, count)]
    result["gold_text"] = [tokenizer(" ".join(annotations), add_special_tokens=False).input_ids[:budget]]
    out = {}
    for arm, part in result.items():
        bodies = [tokenizer.decode(block, skip_special_tokens=True) for block in part]
        visible = [" " + norm(text) + " " for text in bodies]
        hits = [any(" " + norm(fact) + " " in shown for shown in visible) for fact in annotations]
        presentation = a["title_ids"] + part[0]
        if len(part) == 2:
            presentation += separator + part[1]
        assert len(presentation) <= article_limit
        out[arm] = (presentation, hits, sum(map(len, part)))
    return out


def run(a, panel, corpus):
    import torch

    from run_kam_hotpot_pool_scaling import Reader
    from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT
    from run_kam_natural_feasibility import Encoder, batches, measure_answer

    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    rows = panel["rows"][a.shard::a.shards]
    model_dir = a.assets / f"reader_base_{BASE_REVISION}"
    assert digest(model_dir / "model.safetensors") == BASE_WEIGHT
    reader = Reader(model_dir)
    enc = Encoder(a.assets / "encoder", 128)
    with torch.inference_mode():
        needed = {int(id_) for case in rows for id_ in case["evidence"]}
        articles, vectors = article_bank(corpus, needed, reader.tokenizer, enc)
        questions = enc([q["question"] for q in rows], "query")
        prompts, output = {}, []
        separator = reader.tokenizer("…", add_special_tokens=False).input_ids
        for q, vector in zip(rows, questions, strict=True):
            stem = f"Answer the question with a short answer. Question: {q['question']} Evidence:"
            prefix = reader.tokenizer(stem, add_special_tokens=False).input_ids
            cap = min(CAP, (511 - len(prefix)) // 2)
            lists = [select(articles[int(id_)], vector, vectors, reader.tokenizer, facts, cap, separator)
                     for id_, facts in q["evidence"].items()]
            arms = {}
            for mode in ARMS:
                sections = [] if mode == "no_evidence" else [source[mode] for source in lists]
                visible = [vis for _, mentions, _ in sections for vis in mentions]
                prompt = tuple(prefix + [x for piece, _, _ in sections for x in piece] + [reader.tokenizer.eos_token_id])
                assert len(prompt) <= 512
                entry = {"visible_facts": sum(visible), "fact_count": sum(map(len, q["evidence"].values())),
                         "context_body_tokens": sum(n for _, _, n in sections), "input_tokens": len(prompt)}
                prompts.setdefault(prompt, []).append((q["answer"], entry))
                arms[mode] = entry
            output.append({"id": q["id"], "arms": arms})
        for chunk in batches(sorted(prompts, key=len), 32):
            inputs = reader.tokenizer.pad({"input_ids": [list(ids) for ids in chunk]},
                                          padding=True, return_tensors="pt").to("cuda")
            generated = reader.model.generate(**inputs, max_new_tokens=20, do_sample=False)
            texts = reader.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for answer, row in zip(texts, chunk, strict=True):
                for truth, entry in prompts[row]:
                    entry.update(measure_answer(answer, truth))
    result = {"fixture_sha256": digest(a.fixture), "corpus_sha256": panel["corpus_sha256"],
              "reader_weights_sha256": BASE_WEIGHT, "reader_revision": BASE_REVISION,
              "device": torch.cuda.get_device_name(), "shard": a.shard, "shards": a.shards,
              "rows": output, "seconds": perf_counter() - start}
    a.output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("assets", "fixture", "corpus", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    fixture = json.loads(gzip.decompress(a.fixture.read_bytes()))
    corpus = json.loads(a.corpus.read_text())
    assert digest(a.corpus) == fixture["corpus_sha256"]
    run(a, fixture, corpus)


if __name__ == "__main__":
    main()
