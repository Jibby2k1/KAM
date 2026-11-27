#!/usr/bin/env python3
"""Score raw texts, typed date filters, and two-stage semantic routes on vintage records."""

import argparse
from array import array
from collections import Counter
import gzip
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from audit_kam_natural_data import digest, words
from prepare_kam_eurostat_text_routing import COUNTRIES, NAMES, card, document, parse_dates, registry_country, remove_dates
from run_kam_natural_feasibility import Encoder, batches

DEPTH = 64
SHARDS = 4


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def write(path, obj):
    path.write_bytes(gzip.compress(json.dumps(obj, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


def document_text(row):
    return document(*(row[k] for k in ("indicator", "country", "observation", "publication", "value")))


def encode(a, corpus, signature, revisions):
    boundaries = [len(corpus)*i//SHARDS for i in range(SHARDS + 1)]
    lo, hi = boundaries[a.shard:a.shard+2]
    encoder = Encoder(a.assets/"encoder", 128)
    start = perf_counter()
    text = [document_text(source) for source in corpus[lo:hi]]
    order = sorted(range(len(text)), key=lambda i: len(text[i]))
    encoded = encoder([text[i] for i in order], "passage").cpu()
    values = torch.empty_like(encoded)
    values[torch.tensor(order)] = encoded
    torch.cuda.synchronize()
    torch.save({"fixture_sha256": signature, "model_revisions": revisions, "span": [lo, hi], "keys": values,
                "encode_seconds": perf_counter()-start, "device": torch.cuda.get_device_name()}, a.output)


def search_order(data, version_mask):
    rank = data.argsort(dim=1, descending=True, stable=True)[:, :DEPTH].tolist()
    eligible_scores = data.masked_fill(~version_mask, -torch.inf)
    sorted_ids = eligible_scores.argsort(dim=1, descending=True, stable=True)[:, :DEPTH]
    eligible = version_mask.gather(1, sorted_ids).tolist()
    limited = [[d for d, ok in zip(row, valid, strict=True) if ok] for row, valid in zip(sorted_ids.tolist(), eligible, strict=True)]
    return rank, limited


def query_card_text(text, country):
    text = remove_dates(text)
    if country:
        text = re.sub(rf"\b(?:{COUNTRIES[country]}|{country})\b", " ", text)
    return " ".join(text.split())


@torch.inference_mode()
def dense(a, bank, queries, corpus):
    encoder = Encoder(a.assets/"encoder", 128)
    full_cards = [(metric, geo) for metric in NAMES for geo in COUNTRIES]
    cards = [card(m, g) for m, g in full_cards]
    two = list(NAMES)
    no_geo = [f"{NAMES[x]}. Reported monthly percentage." for x in two]
    obs = torch.tensor([r["observation"] for r in corpus], device="cuda")
    published = torch.tensor([r["publication"] for r in corpus], device="cuda")
    raw = [q["text"] for q in queries]
    normalized = [query_card_text(q, registry_country(q)) for q in raw]
    ranking = []
    torch.cuda.synchronize()
    t = perf_counter()
    raw_query, short_query = encoder(raw, "query"), encoder(normalized, "query")
    full_card, fields = encoder(cards, "passage"), encoder(no_geo, "passage")
    flat_score = (raw_query @ full_card.T).tolist()
    metric_score = (short_query @ fields.T).tolist()
    torch.cuda.synchronize()
    query_seconds = perf_counter() - t
    t = perf_counter()
    for slots in batches(list(range(len(queries))), 128):
        selected = torch.tensor(slots, device="cuda")
        allowed_dates = torch.tensor([parse_dates(raw[i]) for i in slots], device="cuda")
        allowed = obs.eq(allowed_dates[:, :1]) & published.le(allowed_dates[:, 1:])
        normal, filtered = search_order(raw_query[selected] @ bank.T, allowed)
        for slot, everyone, recent in zip(slots, normal, filtered, strict=True):
            ranking.append({"id": queries[slot]["id"], "dense_docs": everyone, "dense_docs_dated": recent,
                            "card_scores": flat_score[slot], "metric_scores": metric_score[slot]})
    torch.cuda.synchronize()
    return ranking, perf_counter() - t, query_seconds, full_cards, two


def lexical(corpus, questions):
    raw = [set(words(q["text"])) for q in questions]
    included = set.union(*raw)
    postings = {term: (array("I"), array("I")) for term in included}
    lengths = np.empty(len(corpus))
    t = perf_counter()
    for i, row in enumerate(corpus):
        counts = Counter(words(document_text(row)))
        lengths[i] = counts.total()
        for term, times in counts.items():
            if term in postings:
                idx, freq = postings[term]
                idx.append(i)
                freq.append(times)
    lasts = np.asarray([r["publication"] for r in corpus])
    months = np.asarray([r["observation"] for r in corpus])
    scale = 1.2 * (.25 + .75 * lengths/lengths.mean())
    preprocessed = {term: (np.asarray(ids), np.asarray(tf), np.log1p((len(lengths) - len(ids) + .5) / (len(ids) + .5)))
                    for term, (ids, tf) in postings.items() if ids}
    build = perf_counter()-t
    t = perf_counter()
    rankings = []
    for query, message in zip(raw, questions, strict=True):
        scores = np.zeros(len(lengths))
        for token in sorted(query):
            if token in preprocessed:
                i, tf, weight = preprocessed[token]
                scores[i] += weight * tf * 2.2/(scale[i] + tf)
        ids = np.argsort(-scores, kind="stable")
        mo, co = parse_dates(message["text"])
        eligible = (months == mo) & (lasts <= co)
        rankings.append({"bm25_docs": ids[:DEPTH].tolist(), "bm25_docs_dated": ids[eligible[ids]][:DEPTH].tolist()})
    return rankings, perf_counter()-t, build


def run(a):
    fixture = read(a.fixture)
    corpus = fixture["corpus"]
    signature = digest(a.fixture)
    revisions = read(a.assets/"revisions.json")
    torch.set_float32_matmul_precision("highest")
    if a.mode == "encode":
        encode(a, corpus, signature, revisions)
        return
    groups = sorted([torch.load(p, map_location="cpu", weights_only=True) for p in a.parts], key=lambda p: p["span"])
    assert all(p["fixture_sha256"] == signature and p["model_revisions"] == revisions for p in groups)
    assert groups[0]["span"][0] == 0 and groups[-1]["span"][-1] == len(corpus)
    assert all(groups[i]["span"][1] == groups[i+1]["span"][0] for i in range(len(groups)-1))
    keys = torch.cat([p["keys"] for p in groups]).cuda()
    assert len(keys) == len(corpus)
    sections = {}
    card_labels, metric_labels = None, None
    for split, items in fixture["questions"].items():
        results, score_s, query_s, card_labels, metric_labels = dense(a, keys, items, corpus)
        baseline, lexical_s, build = lexical(corpus, items)
        sections[split] = {"dense_scan_seconds": score_s, "dense_query_seconds": query_s, "bm25_build_seconds": build,
                           "bm25_query_seconds": lexical_s, "rows": [x | y for x, y in zip(results, baseline, strict=True)]}
        print(split, len(items), score_s, query_s, lexical_s, flush=True)
    write(a.output, {"fixture_sha256": signature, "model_revisions": revisions,
                     "card_labels": card_labels, "metric_labels": metric_labels,
                     "device": torch.cuda.get_device_name(), "indexed_records": len(corpus),
                     "encoding_seconds_by_shard": [p["encode_seconds"] for p in groups], "sections": sections})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("encode", "score"))
    for field in ("fixture", "assets", "output"):
        p.add_argument("--" + field, required=True, type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--parts", nargs=4, type=Path)
    run(p.parse_args())


if __name__ == "__main__":
    main()
