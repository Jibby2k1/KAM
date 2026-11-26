#!/usr/bin/env python3
"""Test full-archive retrieval with explicit observation and vintage predicates."""

import argparse
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from prepare_kam_vintage_revisions import SERIES, key
from run_kam_hotpot_independent import read, save
from run_kam_natural_feasibility import Encoder
from run_kam_vintage_revisions import document, read_query

METHODS = ("dense", "dense_cutoff", "dense_observation", "dense_bitemporal")
COHORT = 64


def encode(a):
    start = perf_counter()
    data = read(a.catalog)
    size = len(data["rows"])
    start_id, end_id = a.shard * size // a.shards, (a.shard + 1) * size // a.shards
    enc = Encoder(a.assets / "encoder", a.encode_batch)
    subseq = data["rows"][start_id:end_id]
    docs = [document(rec) for rec in subseq]
    bank = enc(docs, "passage", max_length=64).cpu()
    assert bank.dtype == torch.float32 and torch.isfinite(bank).all()
    meta = {"catalog_sha256": digest(a.catalog), "source_sha256": data["source_sha256"],
            "models": read(a.assets / "revisions.json"), "start": start_id, "end": end_id,
            "device": torch.cuda.get_device_name(), "encoding_seconds": perf_counter() - start}
    torch.save({**meta, "vectors": bank}, a.bank / f"vectors-{a.shard}.pt")
    save(a.output, meta)


def exact_index(rows):
    postings = defaultdict(lambda: ([], []))
    for i, row in enumerate(rows):
        dates, ids = postings[(row["series"], row["observation"])]
        dates.append(row["vintage"])
        ids.append(i)
    assert all(dates == sorted(set(dates)) for dates, _ in postings.values())
    return dict(postings)


def lookup(index, series, observed, cutoff):
    dates, ids = index.get((series, observed), ((), ()))
    i = bisect_right(dates, cutoff) - 1
    return ids[i] if i >= 0 else None


def stable_shortlist(scores):
    values, ids = scores.topk(COHORT + 1, dim=-1, sorted=True)
    boundary_tie = values[:, COHORT - 1].eq(values[:, COHORT]) & values[:, COHORT].isfinite()
    chosen = ids[:, :COHORT].sort(-1).values
    first = scores.gather(1, chosen).argsort(dim=-1, descending=True, stable=True)
    chosen = chosen.gather(1, first)
    if bool(boundary_tie.any()):
        chosen[boundary_tie] = scores[boundary_tie].argsort(dim=-1, descending=True, stable=True)[:, :COHORT]
    return chosen


def retrieval(logits, time_window, observations, cutoff, queried_obs, modes, *, shortlist=False):
    outcomes = {}
    for method in modes:
        mask = None
        if method == "dense_cutoff":
            mask = time_window <= cutoff
        elif method == "dense_observation":
            mask = observations == queried_obs
        elif method == "dense_bitemporal":
            mask = (time_window <= cutoff) & (observations == queried_obs)
        added = logits.masked_fill(~mask, -torch.inf) if mask is not None else logits
        order = stable_shortlist(added) if shortlist else added.argsort(dim=-1, descending=True, stable=True)[:, :COHORT]
        rank_scores = added.gather(1, order)
        outcomes[method] = (order, rank_scores.isfinite())
    return outcomes


@torch.inference_mode()
def evaluate(a):
    begin = perf_counter()
    data, split = read(a.catalog), read(a.fixture)
    assert data["source_sha256"] == split["source_sha256"]
    rows = data["rows"]
    catalog_hash = digest(a.catalog)
    models = read(a.assets / "revisions.json")
    parts = [torch.load(a.bank / f"vectors-{i}.pt", map_location="cpu", weights_only=True) for i in range(a.shards)]
    for i, p in enumerate(parts):
        assert p["models"] == models and p["catalog_sha256"] == catalog_hash
        assert len(rows) * i // a.shards == p["start"] and len(rows) * (i + 1) // a.shards == p["end"]
    bank = torch.cat([p["vectors"] for p in parts]).cuda()
    assert len(bank) == len(rows)
    vintages = torch.tensor([row["vintage"] for row in rows], device="cuda").unsqueeze(0)
    observed = torch.tensor([row["observation"] for row in rows], device="cuda").unsqueeze(0)
    records = [(g, q) for g in split["rows"]["test"] for q in g["questions"]]
    details = [read_query(q["text"]) for _, q in records]
    observations = torch.tensor([item[1] for item in details], device="cuda").unsqueeze(1)
    cutoffs = torch.tensor([item[2] for item in details], device="cuda").unsqueeze(1)
    enc = Encoder(a.assets / "encoder", 128)
    queries = enc([q["text"] for _, q in records], "query")
    aliases = enc([name for name, _, _ in details], "query")
    labels = enc([name for variants in SERIES.values() for name in variants[:2]], "passage")
    which = ((aliases @ labels.T).reshape(len(aliases), len(SERIES), 2).amax(-1)).argmax(-1).tolist()
    idx = exact_index(rows)
    rng = Random(995317)
    output = []
    for start in range(0, len(queries), a.batch):
        sl = slice(start, start + a.batch)
        batch_logits = queries[sl] @ bank.T
        orders = retrieval(batch_logits, vintages, observed, cutoffs[sl], observations[sl], METHODS)
        verified = retrieval(batch_logits, vintages, observed, cutoffs[sl], observations[sl], METHODS, shortlist=True)
        for name in METHODS:
            full, legal = orders[name]
            pruned, seen = verified[name]
            assert torch.equal(legal, seen) and torch.equal(full[legal], pruned[seen])
        mapped = {arm: (indices.tolist(), keep.tolist()) for arm, (indices, keep) in orders.items()}
        for step, j in enumerate(range(start, min(start + a.batch, len(records)))):
            g, q = records[j]
            name, o, v = details[j]
            exact = lookup(idx, g["series"], o, v)
            assert (exact is None) == (q["gold"] is None)
            if exact is not None:
                assert key(rows[exact]) == key(g["candidates"][q["gold"]])
                assert rows[exact]["value"] == q["value"]
            guessed = tuple(SERIES)[which[j]]
            semantic = lookup(idx, guessed, o, v)
            methods = {}
            for mode in METHODS:
                ids, mask = mapped[mode]
                valid = [x for x, keep in zip(ids[step], mask[step], strict=True) if keep]
                selected = valid[0] if valid else None
                found = [i for i in valid if rows[i]["series"] == guessed and rows[i]["observation"] == o and rows[i]["vintage"] <= v]
                pooled_guard = max(found, key=lambda i: rows[i]["vintage"], default=None)
                methods[mode] = {"records": valid, "first": selected, "ranked_oracle": (valid.index(exact) + 1)
                                 if exact in valid else None, "pool_guard": pooled_guard,
                                 "illegal_first": selected is not None and rows[selected]["vintage"] > v,
                                 "value_exact": (rows[selected]["value"] if selected is not None else None) == q["value"]}
            output.append({"id": g["id"], "period": q["period"], "series": g["series"], "style": q["style"],
                           "ground_truth": exact, "semantic_series": guessed, "semantic_index": semantic, "arms": methods})
    timing = {}
    for batch in (1, 16, 64):
        positions = torch.tensor(rng.sample(range(len(queries)), batch), device="cuda")
        for method in METHODS:
            engines = {}
            for label in ("sort", "shortlist"):
                def call():
                    return retrieval(queries[positions] @ bank.T, vintages, observed, cutoffs[positions],
                                     observations[positions], (method,), shortlist=label == "shortlist")
                for _ in range(5):
                    call()
                torch.cuda.synchronize()
                points = []
                for _ in range(5):
                    t = perf_counter()
                    for _ in range(16):
                        call()
                    torch.cuda.synchronize()
                    points.append((perf_counter() - t) * 1000 / 16)
                engines[label] = {"batch_ms": median(points), "repetitions": points}
            timing[f"{batch}:{method}"] = engines
    save(a.output, {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                    "fixture_sha256": digest(a.fixture), "catalog_sha256": catalog_hash,
                    "corpus_records": len(rows), "stage_vector_bytes": bank.numel() * bank.element_size(),
                    "models": models, "selected_depth": COHORT, "rows": output, "timing": timing,
                    "elapsed_seconds": perf_counter() - begin})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("encode", "eval"))
    for name in ("assets", "fixture", "catalog", "bank", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--encode-batch", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    if a.mode == "encode":
        encode(a)
    else:
        with torch.inference_mode():
            evaluate(a)


if __name__ == "__main__":
    main()
