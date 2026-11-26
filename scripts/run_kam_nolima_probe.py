#!/usr/bin/env python3
"""Evaluate exact needle recall under literal and associative NoLiMa questions."""

import argparse
import json
from pathlib import Path
from random import Random
import re
from time import perf_counter

import torch

from audit_kam_natural_data import digest, rank, stable, words
from run_kam_hotpot_head_to_head import route
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader, pair_selection
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, measure_answer

DEPTHS = (16, 64, 256)


def render(template, letters, character):
    return re.sub(r"\{(\d+|CHAR)\}", lambda m: character if m[1] == "CHAR" else letters[int(m[1]) - 1], template)


def episodes(a):
    sources = {p: read(getattr(a, p)) for p in ("minimal", "direct", "distractors")}
    direct = {r["id"]: r for r in sources["direct"]}
    distractors = {r["id"]: r for r in sources["distractors"]}
    books = [p.read_text().split() for p in a.books]
    panels = []
    for series in sources["minimal"]:
        template_id = series["id"]
        direct_row = direct[template_id]
        distract = distractors[template_id]
        for key, test in series["tests"].items():
            if key not in direct_row["tests"]:
                continue
            args = test["input_args"]
            dargs = direct_row["tests"][key]["input_args"]
            if render(direct_row["needle"], dargs, "TEST") != render(series["needle"], args, "TEST"):
                continue
            assert distract["tests"][key]["input_args"] == args
            for replica, book in enumerate(books):
                id_ = f"{template_id}/{key}/{replica}"
                seed = int(stable("kam-nolima-v1:" + id_)[:12], 16)
                rng = Random(seed)
                n = max(DEPTHS)
                start = rng.randrange(len(book) - n * 96)
                chunks = [" ".join(book[start + 96 * j:start + 96 * (j + 1)]) for j in range(n)]
                background_words = set(words(" ".join(chunks)))
                available = [name for name in series["character_set"] if name.casefold() not in background_words]
                if not available:
                    continue
                character = rng.choice(available)
                needle = render(series["needle"], args, character)
                assert needle == render(direct_row["needle"], dargs, character)
                chosen = rng.randrange(min(DEPTHS))
                bad = rng.choice([i for i in range(min(DEPTHS)) if i != chosen])
                offsets = (rng.randrange(12, 48), rng.randrange(12, 48))

                def insert(text, snippet, offset):
                    units = text.split()
                    return " ".join([*units[:offset], snippet, *units[offset:]])

                chunks[chosen] = insert(chunks[chosen], needle, offsets[0])
                for method, prototype in {"direct": direct_row["questions"]["direct"], **series["questions"]}.items():
                    q = render(prototype, dargs if method == "direct" else args, character)
                    rendered = render(distract["distractors"][method], args, character) if method in distract["distractors"] else None
                    inserted = insert(chunks[bad], rendered, offsets[1]) if rendered else None
                    panels.append({"id": f"{id_}/{method}", "base": id_, "kind": method, "template_id": template_id,
                                   "question": q, "answer": character, "chunks": chunks, "replacement": inserted,
                                   "needle": chosen, "distractor": bad, "offsets": offsets})
    return panels


@torch.inference_mode()
def evaluate(a, panels):
    torch.set_float32_matmul_precision("highest")
    enc = Encoder(a.assets / "encoder", 128)
    reader = Reader(a.reader)
    cfg = read(a.config)
    adapter, original = restore_checkpoint(a.baseline, a.config, cfg, None, None,
                                           weights_sha256=a.baseline_sha256)
    vocabulary = list(dict.fromkeys(
        text for item in panels for text in item["chunks"] + ([item["replacement"]] if item["replacement"] else [])
    ))
    bank = enc(vocabulary, "passage")
    id_by_text = dict(zip(vocabulary, range(len(vocabulary)), strict=True))
    qbank = enc([item["question"] for item in panels], "query")
    pairs = {depth: torch.triu_indices(depth, depth, 1, device="cuda") for depth in DEPTHS}
    requests, output = [], []
    for i, item in enumerate(panels):
        question, q = item["question"], qbank[i:i+1]
        adapted = adapter(q)
        scenes = {"clean": item["chunks"]}
        if item["replacement"]:
            noise = list(item["chunks"])
            noise[item["distractor"]] = item["replacement"]
            scenes["lexical_distractor"] = noise
        result = {"id": item["id"], "base": item["base"], "kind": item["kind"], "template": item["template_id"],
                  "needle": item["needle"], "distractor": item["distractor"], "scenes": {}}
        for name, corpus in scenes.items():
            info = {}
            for depth in DEPTHS:
                text = corpus[:depth]
                sample = {"question": question, "context": [("Excerpt", [doc]) for doc in text]}
                selected = bank[torch.tensor([id_by_text[t] for t in text], device="cuda")].unsqueeze(0)
                affinity = (q @ selected[0].T)[0]
                top2 = affinity.argsort(descending=True, stable=True)[:2].tolist()
                routes, _ = route(enc, adapter, [sample], q, selected, include_frozen=False)
                arms = {
                    "bm25": rank(question, text, 2),
                    "dense": top2,
                    "learned_single_shot": routes["learned_single_shot"][0],
                    "learned_sequential": routes["learned_sequential"][0],
                    "one_pass_pair_without_title_links": pair_selection(adapted, selected, None, original, pairs[depth])[0].tolist(),
                }
                found = {}
                for arm, slots in arms.items():
                    record = {"needle": item["needle"] in slots, "distractor": item["distractor"] in slots,
                              "slots": slots}
                    found[arm] = record
                    requests.append((reader.prompt(sample, slots), record, item["answer"]))
                info[depth] = found
            result["scenes"][name] = info
        gold = {"question": question, "context": [("Excerpt", [item["chunks"][item["needle"]]])]}
        result["oracle_reader"] = {}
        result["closed_reader"] = {}
        requests.append((reader.prompt(gold, [0]), result["oracle_reader"], item["answer"]))
        requests.append((reader.prompt(gold, []), result["closed_reader"], item["answer"]))
        output.append(result)
    texts = list(dict.fromkeys(t for t, _, _ in requests))
    decoded = dict(zip(texts, reader.answers(texts), strict=True))
    for text, data, gold in requests:
        prediction = decoded[text]
        data["em"] = measure_answer(prediction, gold)["em"]
        data["target_mentioned"] = gold.casefold() in words(prediction)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "baseline", "config", "minimal", "direct", "distractors", "reader", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--books", type=Path, nargs=2, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    full = episodes(a)
    panels = full[a.shard::a.shards]
    rows = evaluate(a, panels)
    save(a.output, {"n_total": len(full), "inputs": {n: digest(p) for n, p in
                    [(n, getattr(a, n)) for n in ("minimal", "direct", "distractors")] +
                    [(f"book_{i}", p) for i, p in enumerate(a.books)]},
                    "reader_sha256": digest(a.reader / "files.sha256"), "device": torch.cuda.get_device_name(),
                    "elapsed_seconds": perf_counter() - start, "rows": rows})


if __name__ == "__main__":
    main()
