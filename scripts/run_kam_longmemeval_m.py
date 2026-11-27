#!/usr/bin/env python3
"""Retrieve dated LongMemEval-M sessions and turn-level author evidence."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest

BANK_SHARDS = 8
SIZES = (64, 128, 256, 512)
METHODS = ("dense", "user", "assistant", "length_corrected", "dense_recent", "asof_dense", "recent_only")
SESSION_TOP = 20

def iter_bank(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def subdivide(a):
    from contextlib import ExitStack

    manifest = json.loads(a.fixture.read_text())
    assert digest(a.bank) == manifest["session_bank_sha256"]
    a.output.mkdir(exist_ok=False)
    paths = [a.output / f"sessions-{i}.jsonl.gz" for i in range(BANK_SHARDS)]
    with ExitStack() as stack:
        writers = []
        for path in paths:
            raw = stack.enter_context(path.open("wb"))
            writers.append(stack.enter_context(gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="", compresslevel=4)))
        count = 0
        for item in iter_bank(a.bank):
            i = item["id"]
            assert i == count
            writer = writers[i % BANK_SHARDS]
            writer.write((json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
            count += 1
    assert count == manifest["summary"]["distinct_sessions"]
    summary = {"source_sha256": manifest["source_sha256"], "fixture_sha256": digest(a.fixture),
               "bank_sha256": manifest["session_bank_sha256"], "shards": [digest(p) for p in paths]}
    (a.output / "bank-index.json").write_text(json.dumps(summary, separators=(",", ":")) + "\n")


def chunks(turns):
    for i, (role, text) in enumerate(turns):
        words = text.split()
        for start in range(0, len(words), 96):
            yield i, role == "user", role == "assistant", role + ": " + " ".join(words[start:start+128])
            if start + 128 >= len(words):
                break


@torch.inference_mode()
def embed(a, keys, fixture):
    from run_kam_natural_feasibility import Encoder

    assert digest(a.bank) == keys["shards"][a.shard]
    encoder = Encoder(a.assets / "encoder", a.batch)
    piece, ids, descriptors = [], [], []
    offsets = [0]
    start = perf_counter()
    output, user, assistant, turns = [], [], [], []

    def encode_group():
        if not piece:
            return
        # Bucket by text length inside each buffer to reduce padding.
        order = sorted(range(len(piece)), key=lambda k: len(piece[k]))
        values = encoder([piece[i] for i in order], "passage").cpu()
        arranged = torch.empty_like(values)
        arranged[order] = values
        output.append(arranged)
        piece.clear()

    for r in iter_bank(a.bank):
        assert r["id"] % BANK_SHARDS == a.shard
        ids.append(r["id"])
        c = list(chunks(r["turns"]))
        offsets.append(offsets[-1] + len(c))
        for turn, u, s, text in c:
            turns.append(turn)
            user.append(u)
            assistant.append(s)
            piece.append(text)
        if len(piece) >= 16384:
            encode_group()
        descriptors.append(len(r["turns"]))
    encode_group()
    torch.cuda.synchronize()
    memory = torch.cat(output, 0)
    assert len(memory) == offsets[-1] == len(user) == len(assistant) == len(turns)
    torch.save({"ids": ids, "offsets": offsets, "vectors": memory, "turn_counts": descriptors,
                "user": torch.tensor(user, dtype=torch.bool), "assistant": torch.tensor(assistant, dtype=torch.bool),
                "turns": torch.tensor(turns, dtype=torch.int16), "shard": a.shard,
                "source_sha256": fixture["source_sha256"], "bank_sha256": keys["bank_sha256"],
                "part_sha256": keys["shards"][a.shard], "fixture_sha256": keys["fixture_sha256"],
                "models": json.loads((a.assets / "revisions.json").read_text()),
                "embedding_seconds": perf_counter() - start, "device": torch.cuda.get_device_name()}, a.output)


def by_id(embeddings, keys, fixture, revisions):
    lookup = {}
    for path in embeddings:
        b = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
        assert b["source_sha256"] == fixture["source_sha256"] and b["bank_sha256"] == keys["bank_sha256"]
        assert b["fixture_sha256"] == keys["fixture_sha256"] and b["part_sha256"] == keys["shards"][b["shard"]]
        assert b["models"] == revisions and b["vectors"].dtype == torch.float32
        for i, k in enumerate(b["ids"]):
            l, r = b["offsets"][i:i+2]
            assert k not in lookup
            lookup[k] = (b["vectors"][l:r], b["user"][l:r], b["assistant"][l:r], b["turns"][l:r], b["turn_counts"][i])
    assert set(lookup) == set(range(fixture["summary"]["distinct_sessions"]))
    return lookup


def nested_positions(item):
    ids = item["source_session_ids"]
    target = set(item["gold_ids"])
    required, background = [], []
    for slot, sid in enumerate(ids):
        if sid in target:
            required.append(slot)
        else:
            key = sha256((item["id"] + ":" + sid + ":" + str(slot)).encode()).digest()
            background.append((key, slot))
    background.sort()
    return {n: sorted(required + [j for _, j in background[:max(0, n - len(required))]]) for n in SIZES}


def session_ranks(values, subset, ids):
    values = values[subset]
    ranked = values.argsort(descending=True, stable=True).tolist()
    finite = values.isfinite().tolist()
    unique, chosen = set(), []
    for j in ranked:
        i = subset[j]
        sid = ids[i]
        if finite[j] and sid not in unique:
            unique.add(sid)
            chosen.append(i)
            if len(chosen) == SESSION_TOP:
                break
    return chosen


def turn_order(scores, slots, turns, session_pool, k=20):
    keep = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
    for s in session_pool:
        keep |= slots == s
    rank = scores.masked_fill(~keep, -torch.inf).argsort(descending=True, stable=True)
    items, seen = [], set()
    indices = rank[keep[rank]]
    pairs = torch.stack((slots[indices], turns[indices]), dim=1).tolist()
    for slot, turn in pairs:
        key = (slot, turn)
        if key not in seen:
            items.append([slot, turn])
            seen.add(key)
            if len(items) >= k:
                break
    return items


@torch.inference_mode()
def evaluate(a, keys, fixture):
    from run_kam_natural_feasibility import Encoder

    start = perf_counter()
    revisions = json.loads((a.assets / "revisions.json").read_text())
    lookup = by_id(a.vectors, keys, fixture, revisions)
    enc = Encoder(a.assets / "encoder", 128)
    cases = fixture["rows"][a.shard::a.shards]
    qv = enc([r["question"] for r in cases], "query")
    rows = []
    for q, item in zip(qv, cases, strict=True):
        refs = [lookup[i] for i in item["slots"]]
        lens = torch.tensor([len(ref[0]) for ref in refs], device="cuda")
        dates = torch.tensor(item["dates"], device="cuda")
        c = len(refs)
        assert len(dates) == c == len(item["source_session_ids"])
        slots = torch.arange(c, device="cuda").repeat_interleave(lens)
        vector, user, assistant, turns = (torch.cat([ref[k] for ref in refs], 0).cuda() for k in range(4))
        query_t = item["time"]
        sim = vector @ q
        assert len(slots) == len(sim)
        finite = torch.full((c,), -torch.inf, device="cuda")
        high = finite.scatter_reduce(0, slots, sim, reduce="amax")
        user_only = finite.scatter_reduce(0, slots, sim.masked_fill(~user, -torch.inf), reduce="amax")
        assistant_only = finite.scatter_reduce(0, slots, sim.masked_fill(~assistant, -torch.inf), reduce="amax")
        scores = sim.sort().values
        tail_fraction = (len(scores) - torch.searchsorted(scores, high) + 1) / (len(scores) + 1)
        corrected = -(tail_fraction * lens).clamp(max=1).clamp_min(1 / (len(scores) + 1)).log()
        corrected.masked_fill_(lens == 0, -torch.inf)
        asof = dates <= query_t
        recent = 0.025 * ((dates - query_t).clamp_max(0).float() / (30 * 1440)).exp()
        modes = {"dense": high, "user": user_only, "assistant": assistant_only, "length_corrected": corrected,
                 "dense_recent": high + recent, "asof_dense": high.masked_fill(~asof, -torch.inf),
                 "recent_only": dates.double().masked_fill(~asof, -torch.inf)}
        pieces = {}
        for k, subset in nested_positions(item).items():
            pieces[str(k)] = {arm: session_ranks(values, subset, item["source_session_ids"]) for arm, values in modes.items()}
        full = {arm: session_ranks(values, list(range(c)), item["source_session_ids"]) for arm, values in modes.items()}
        picked_turns = {arm: turn_order(sim, slots, turns.long(), ranks[:5])
                        for arm, ranks in full.items() if arm in ("dense", "user", "length_corrected")}
        rows.append({"id": item["id"], "session_count": c, "chunk_count": len(sim), "full": full,
                     "nested": pieces, "turns_from_top5": picked_turns})
    out = {"fixture_sha256": keys["fixture_sha256"], "source_sha256": fixture["source_sha256"], "models": revisions,
           "bank_sha256": keys["bank_sha256"], "device": torch.cuda.get_device_name(), "shard": a.shard,
           "shards": a.shards, "elapsed_seconds": perf_counter() - start, "rows": rows}
    a.output.write_bytes(gzip.compress(json.dumps(out, separators=(",", ":")).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("split", "embed", "evaluate"))
    for key in ("assets", "bank", "bank-index", "fixture", "output"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--vectors", nargs=BANK_SHARDS, type=Path)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--batch", type=int, default=192)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    if a.mode == "split":
        subdivide(a)
        return
    torch.set_float32_matmul_precision("highest")
    keys = json.loads(a.bank_index.read_text())
    fixture = json.loads(a.fixture.read_text())
    assert keys["fixture_sha256"] == digest(a.fixture) and keys["bank_sha256"] == fixture["session_bank_sha256"]
    (embed if a.mode == "embed" else evaluate)(a, keys, fixture)


if __name__ == "__main__":
    main()
