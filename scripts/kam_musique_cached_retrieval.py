#!/usr/bin/env python3
"""Compare cached MuSiQue retrievers with a common live Llama endpoint decision."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from zipfile import ZipFile

from audit_kam_natural_data import stable, words
from eval_kam_pretrained_endpoint_selection import literal, prompts
from kam_adapter_preservation import digest, key
from kam_math_kl_protocol import test_runs, write_json
from kam_musique_live_routes import prepare as prepare_pairs
from prepare_kam_native_musique import MEMBERS, canonical_paragraph

PROTOCOL = "kam-musique-cached-retrieval-v1"
ROUTES = ("bm25", "dense", "hybrid")
GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
ENCODER = "intfloat/e5-small-v2"
REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
RANK_CONSTANT = 60
ENCODER_LIMIT = 256
READER_LIMIT = 512
REPEATS = 3
ENCODER_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.txt")


def display(doc):
    return doc["title"] + ". " + doc["body"]


def prepare(source, priors, title_study, encoder_files, work):
    title_rows = json.loads(title_study.read_text())
    reserved_ids = {r["id"] for split in ("fit", "heldout") for r in title_rows[split]["records"]}
    work.mkdir(parents=True, exist_ok=True)
    prior_pairs = [pair for prior in priors for pair in json.loads(prior.read_text())]
    old_families = {pair["family"] for pair in prior_pairs}
    with TemporaryDirectory() as directory:
        scratch = Path(directory)
        reservation = scratch / "prior.json"
        write_json(reservation, prior_pairs)
        prepare_pairs(source, reservation, scratch, protocol=PROTOCOL, reserved_ids=reserved_ids, pair_cap=96,
                      hold_families=False)
        pairs = json.loads((scratch / "panel.json").read_text())
    targets = {q["id"]: (pair, q) for pair in pairs for q in pair["questions"]}
    assert len(targets) == 2 * len(pairs)
    pools = {}
    with ZipFile(source) as archive:
        for split, member in MEMBERS.items():
            with archive.open(member) as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["id"] not in targets:
                        continue
                    pair, question = targets[row["id"]]
                    assert pair["split"] == split and question["question"] == row["question"]
                    earlier = {q["evidence"][0]["id"]: display(q["evidence"][0]) for q in pair["questions"]}
                    excluded = {p["id"] for q in pair["questions"] for p in q["evidence"]}
                    native = {}
                    for paragraph in row["paragraphs"]:
                        ident = stable("\x1f".join(canonical_paragraph(paragraph)))
                        if not paragraph["is_supporting"] and ident not in excluded:
                            native[ident] = paragraph["title"] + ". " + paragraph["paragraph_text"]
                    docs = earlier | native
                    order = sorted(docs, key=lambda ident: stable(PROTOCOL + row["id"] + ident))
                    pools[row["id"]] = {"ids": order, "texts": [docs[ident] for ident in order]}
    assert pools.keys() == targets.keys()
    pairs = [p for p in pairs if all(len(pools[q["id"]]["ids"]) >= 18 for q in p["questions"])]
    assert len(pairs) >= 12
    cases = [{"pair": p["id"], "family": p["family"], "split": p["split"], "prior_family": p["family"] in old_families,
              "questions": [q["id"] for q in p["questions"]]} for p in pairs]
    pools = {q["id"]: pools[q["id"]] for pair in pairs for q in pair["questions"]}
    fixture = work / "panel.json"
    write_json(fixture, {"pairs": pairs, "pools": pools})
    files = json.loads(encoder_files.read_text())
    assert set(files) == set(ENCODER_FILES)
    write_json(work / "manifest.json", {
        "protocol": PROTOCOL, "routes": ROUTES, "gpu": GPU, "source_sha256": digest(source),
        "prior_pair_sha256": [digest(p) for p in priors], "prior_title_study_sha256": digest(title_study),
        "panel_sha256": digest(fixture), "cases": cases, "pool_ids": {q: p["ids"] for q, p in pools.items()},
        "encoder": ENCODER, "encoder_revision": REVISION, "encoder_files": files, "encoder_limit": ENCODER_LIMIT,
        "reader_limit": READER_LIMIT, "rank_constant": RANK_CONSTANT, "repeats": REPEATS, "deadline_ms": 60,
    })


def load_frozen(work, assets):
    manifest, panel = work / "manifest.json", work / "panel.json"
    config = json.loads(manifest.read_text())
    assert config["protocol"] == PROTOCOL and tuple(config["routes"]) == ROUTES
    assert config["panel_sha256"] == digest(panel)
    assert config["encoder_revision"] == json.loads((assets / "revisions.json").read_text())[ENCODER] == REVISION
    actual = {name: digest(assets / "encoder" / name) for name in ENCODER_FILES}
    assert config["encoder_files"] == actual
    return config, json.loads(panel.read_text())


def build_cache(root, work, assets):
    import torch
    from transformers import AutoTokenizer
    from run_kam_natural_feasibility import Encoder

    config, data = load_frozen(work, assets)
    assert torch.cuda.get_device_name() == config["gpu"] == GPU
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
    excluded = []
    for pair in data["pairs"]:
        # Match the previous visibility rule and also check the exact displayed passages.
        try:
            prompts(tokenizer, [pair])
        except AssertionError as error:
            if str(error) not in {q["id"] for q in pair["questions"]}:
                raise
            excluded.append(pair["id"])
            continue
        if not all(visible_gold(tokenizer, q) for q in pair["questions"]):
            excluded.append(pair["id"])
    used = {q["id"] for pair in data["pairs"] if pair["id"] not in excluded for q in pair["questions"]}
    assert len(used) >= 24
    passages = [text for q, pool in data["pools"].items() if q in used for text in pool["texts"]]
    unique = list(dict.fromkeys(passages))
    encoder = Encoder(assets / "encoder", batch=64)
    embeddings = dict(zip(unique, encoder(unique, "passage", max_length=ENCODER_LIMIT).cpu(), strict=True))
    cache = work / "passages.pt"
    torch.save({q: torch.stack([embeddings[text] for text in pool["texts"]])
                for q, pool in data["pools"].items() if q in used}, cache)
    write_json(work / "cache.json", {"manifest_sha256": digest(work / "manifest.json"), "cache_sha256": digest(cache),
                                    "excluded_pairs": sorted(excluded), "unique_passages": len(unique),
                                    "queries": len(used), "cache_bytes": cache.stat().st_size})


def clip(tokenizer, text):
    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=READER_LIMIT, truncation=True)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def visible_gold(tokenizer, question):
    first, second = [clip(tokenizer, display(p)) for p in question["evidence"]]
    return (literal(question["anchor"], first) and literal(question["anchor"], second) and
            any(literal(value, second) for value in question["answers"]))


class BM25Pool:
    def __init__(self, passages):
        self.counts = [Counter(words(text)) for text in passages]
        frequency = Counter(term for counts in self.counts for term in counts)
        lengths = [sum(counts.values()) for counts in self.counts]
        average = sum(lengths) / len(lengths) or 1
        self.norm = [1.2 * (.25 + .75 * length / average) for length in lengths]
        self.idf = {term: math.log1p((len(lengths) - n + .5) / (n + .5)) for term, n in frequency.items()}

    def rank(self, query, order):
        terms = [(term, self.idf[term]) for term in sorted(set(words(query))) if term in self.idf]
        values = [math.fsum(idf * (tf := counts[term]) * 2.2 / (tf + norm) for term, idf in terms)
                  for counts, norm in zip(self.counts, self.norm, strict=True)]
        return sorted(order, key=lambda i: -values[i])


def fuse(lexical, dense, order):
    ranks = {index: 1 / (RANK_CONSTANT + rank) for rank, index in enumerate(lexical, 1)}
    for rank, index in enumerate(dense, 1):
        ranks[index] += 1 / (RANK_CONSTANT + rank)
    return sorted(order, key=lambda index: -ranks[index])


def evaluate(root, work, assets, index):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import CONTEXT, next_logits, pad
    from run_kam_natural_feasibility import Encoder

    config, data = load_frozen(work, assets)
    selection = work / "selection.json"
    chosen = json.loads(selection.read_text())
    assert (chosen["selected_recipe"], chosen["selected_kl_weight"]) == ("lower_lr_kl05", .5)
    run = test_runs(chosen["selected_recipe"])[index]
    model, tokenizer, stack, metadata = load_checkpoint(root, *run)
    assert torch.cuda.get_device_name() == config["gpu"] == GPU
    encoder = Encoder(assets / "encoder", batch=1)
    cache_meta = json.loads((work / "cache.json").read_text())
    assert cache_meta["manifest_sha256"] == digest(work / "manifest.json")
    assert cache_meta["cache_sha256"] == digest(work / "passages.pt")
    dense_cache = torch.load(work / "passages.pt", map_location="cuda", weights_only=True)
    excluded = set(cache_meta["excluded_pairs"])
    assert isinstance(model.lm_head, torch.nn.Linear) and model.lm_head.bias is None
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
    assert all(len(ids) == 1 for ids in letters) and letters[0] != letters[1]
    projection = model.lm_head.weight.detach().index_select(0, torch.tensor([ids[0] for ids in letters], device="cuda"))

    def route(method, bm25, vectors, order, query, clipped_pool, endpoints):
        start = perf_counter_ns()
        if method != "dense":
            lexical = bm25.rank(query, order)
        if method != "bm25":
            encoded = encoder([query], "query", max_length=ENCODER_LIMIT)[0]
            scores = (vectors @ encoded).tolist()
            dense = sorted(order, key=lambda i: -scores[i])
        ranking = fuse(lexical, dense, order) if method == "hybrid" else (lexical if method == "bm25" else dense)
        retrieve_ms = (perf_counter_ns() - start) / 1e6
        message = ("Select the passage that answers the question. Use earlier evidence to identify the right entity "
                   "when it is available. Reply with A or B.\n\nQuestion: " + query +
                   "\n\nEarlier passage:\n" + clipped_pool[ranking[0]] +
                   "\n\nPassage A:\n" + endpoints[0] + "\n\nPassage B:\n" + endpoints[1] + "\n\nChoice:")
        text = tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                             add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) <= CONTEXT
        x, mask = pad([ids], tokenizer.pad_token_id)
        logits, _ = next_logits(model, stack, x, mask, projection=projection)
        a, b = logits[0].tolist()
        return (ranking, b - a, len(ids)), retrieve_ms

    tasks = []
    for pair in data["pairs"]:
        if pair["id"] in excluded:
            continue
        endpoints = [clip(tokenizer, display(q["evidence"][1])) for q in pair["questions"]]
        for j, q in enumerate(pair["questions"]):
            pool = data["pools"][q["id"]]
            assert pool["ids"] == config["pool_ids"][q["id"]]
            first = pool["ids"].index(q["evidence"][0]["id"])
            vectors = dense_cache[q["id"]]
            assert len(pool["texts"]) == len(vectors) == len(set(pool["ids"]))
            tasks.append((pair, q, j, BM25Pool(pool["texts"]), vectors, first, endpoints,
                          [clip(tokenizer, text) for text in pool["texts"]]))
    assert len(tasks) == cache_meta["queries"] == len(dense_cache)
    for warm in (tasks[0], tasks[len(tasks) // 2], tasks[-1]):
        _, q, _, bm25, vectors, _, endpoints, clipped_pool = warm
        for method in ROUTES:
            route(method, bm25, vectors, range(len(vectors)), q["question"], clipped_pool, endpoints)

    records = []
    for i, (pair, q, j, bm25, vectors, first, endpoints, clipped_pool) in enumerate(tasks):
        for reverse in (False, True):
            offered = endpoints[::-1] if reverse else endpoints
            expected = j ^ reverse
            for slot, order in enumerate((range(len(vectors)), range(len(vectors) - 1, -1, -1))):
                record = {"pair": pair["id"], "family": pair["family"], "question": q["id"],
                          "reverse": reverse, "expected": expected, "earlier_slot": slot, "pool_size": len(order)}
                trials = {method: [] for method in ROUTES}
                for repeat in range(REPEATS):
                    shift = (i + 2 * reverse + slot + repeat) % len(ROUTES)
                    for method in ROUTES[shift:] + ROUTES[:shift]:
                        torch.cuda.synchronize()
                        start = perf_counter_ns()
                        result, retrieval = route(method, bm25, vectors, order, q["question"], clipped_pool, offered)
                        torch.cuda.synchronize()
                        trials[method].append((result, retrieval, (perf_counter_ns() - start) / 1e6))
                for method, runs in trials.items():
                    ranking, margin, tokens = runs[0][0]
                    assert all(trial[0] == runs[0][0] for trial in runs)
                    record[method] = {"first_rank": ranking.index(first) + 1, "first_correct": ranking[0] == first,
                                      "correct": margin != 0 and int(margin > 0) == expected, "margin": margin,
                                      "tokens": tokens, "retrieve_ms": median(t[1] for t in runs),
                                      "median_ms": median(t[2] for t in runs)}
                records.append(record)
    write_json(work / f"routes-{key(*run)}.json", {"metadata": {**metadata, "gpu": GPU, "protocol": PROTOCOL,
               "manifest_sha256": digest(work / "manifest.json"), "selection_sha256": digest(selection),
               "cache": cache_meta, "excluded_pairs": cache_meta["excluded_pairs"]}, "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("source", "title_study", "encoder_files", "work"):
        prep.add_argument(name, type=Path)
    prep.add_argument("prior", type=Path, nargs="+")
    for name in ("cache", "evaluate"):
        gpu = commands.add_parser(name)
        for item in ("root", "work", "assets"):
            gpu.add_argument(item, type=Path)
        if name == "evaluate":
            gpu.add_argument("index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.prior, args.title_study, args.encoder_files, args.work)
    elif args.command == "cache":
        build_cache(args.root, args.work, args.assets)
    else:
        evaluate(args.root, args.work, args.assets, args.index)
