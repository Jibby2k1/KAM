#!/usr/bin/env python3
"""Test whether indexing additional parts of long BRIGHT documents improves retrieval."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from math import ceil
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import torch
from torch.nn.functional import normalize

from audit_kam_natural_data import digest
from run_kam_bright_bm25 import corpus
from run_kam_bright_transfer import LIMIT, order, rrf
from run_kam_natural_feasibility import Encoder, batches

LIMIT_TOKENS = 192
MAX_VIEWS = 4
SHAM_SEEDS = (839723, 839731, 839749)


def windows(ids, cap):
    if len(ids) <= cap:
        return []
    steps = min(MAX_VIEWS-1, ceil((len(ids)-cap)/cap))
    starts = [(len(ids)-cap)*i//steps for i in range(1, steps+1)]
    return [(ids[pos:pos+cap], i == steps) for i, pos in enumerate(starts, 1)]


def encode(a, fixture, revisions):
    part = fixture["parts"][a.shard]
    source = a.inputs / f"corpus-{a.shard}.jsonl.gz"
    assert digest(source) == part["sha256"]
    docs = list(corpus(source))
    assert len(docs) == part["high"] - part["low"]
    enc = Encoder(a.assets / "encoder", 128)
    tok = enc.tokenizer
    prefix = tok("passage:", add_special_tokens=False).input_ids
    probe = tok("passage: Sample document", return_special_tokens_mask=True)
    normal = [i for i, special in enumerate(probe.special_tokens_mask) if not special]
    assert normal == list(range(normal[0], normal[-1] + 1))
    before, after = probe.input_ids[:normal[0]], probe.input_ids[normal[-1]+1:]
    assert len(before) + len(after) == tok.num_special_tokens_to_add()
    cap = LIMIT_TOKENS - len(prefix) - len(before) - len(after)
    job_start = perf_counter()
    inputs, numbers, owners, final = [], [], [], []
    for texts in batches([r["content"] for r in docs], 256):
        pieces = tok(["passage: " + text for text in texts], add_special_tokens=False, verbose=False).input_ids
        for ids in pieces:
            assert ids[:len(prefix)] == prefix
            row = len(numbers)
            original = ids[len(prefix):]
            numbers.append(len(original))
            for selected, end in windows(original, cap):
                inputs.append(before + prefix + selected + after)
                owners.append(part["low"] + row)
                final.append(end)
    with torch.inference_mode():
        chunks = []
        for rows in batches(inputs, 128):
            padded = tok.pad({"input_ids": rows}, padding=True, return_tensors="pt").to("cuda")
            output = enc.model(**padded).last_hidden_state
            chunks.append(normalize((output * padded.attention_mask.unsqueeze(-1)).sum(1), dim=-1).cpu())
    torch.cuda.synchronize()
    assert sum(final) == sum(n > cap for n in numbers)
    torch.save({"part": part, "fixture_sha256": digest(a.inputs/"fixture.json"), "revisions": revisions,
                "owner": torch.tensor(owners, dtype=torch.long), "tail": torch.tensor(final, dtype=torch.bool),
                "lengths": numbers, "cap": cap, "vectors": torch.cat(chunks) if chunks else torch.empty(0, enc.model.config.hidden_size),
                "seconds": perf_counter()-job_start, "device": torch.cuda.get_device_name()}, a.output)


def load_records(paths, field, fixture, revisions, fixture_hash):
    stored = sorted([torch.load(p, map_location="cpu", weights_only=True) for p in paths], key=lambda v: v[field]["low"])
    assert [r[field] for r in stored] == fixture["parts"]
    assert all(r["fixture_sha256"] == fixture_hash and r["revisions"] == revisions for r in stored)
    return stored


def permuted_owners(owners, lengths, seed):
    group = defaultdict(list)
    for doc, views in Counter(owners).items():
        group[views, min(lengths[doc] // 128, 31)].append(doc)
    rng = Random(seed)
    targets = {}
    for members in group.values():
        rng.shuffle(members)
        targets.update(zip(members, members[1:] + members[:1], strict=True))
    moved = [targets[i] for i in owners]
    assert Counter(moved) == Counter(owners)
    return moved, sum(original != replacement for original, replacement in targets.items())


def search(a, fixture, revisions):
    fixture_hash = digest(a.inputs/"fixture.json")
    head = load_records(a.head, "provenance", fixture, revisions, fixture_hash)
    extra = load_records(a.extra, "part", fixture, revisions, fixture_hash)
    lexical = json.loads(gzip.decompress(a.lexical.read_bytes()))
    assert lexical["fixture_sha256"] == fixture_hash
    assert [q["id"] for q in fixture["rows"]] == [r["id"] for r in lexical["rows"]]
    cases = fixture["rows"]
    docs = torch.cat([r["vectors"] for r in head]).cuda()
    windows_bank = torch.cat([r["vectors"] for r in extra]).cuda()
    owners = torch.cat([r["owner"] for r in extra]).cuda()
    tail = torch.cat([r["tail"] for r in extra]).cuda()
    lengths = [n for part in extra for n in part["lengths"]]
    assert len(docs) == len(lengths) == fixture["documents"]
    boundaries = {r["cap"] for r in extra}
    assert len(boundaries) == 1
    cap, = boundaries
    assert len(owners) == len(windows_bank) and int(tail.sum()) == sum(n > cap for n in lengths)
    assert owners[tail].unique().numel() == int(tail.sum())
    enc = Encoder(a.assets/"encoder", 128)
    with torch.inference_mode():
        queries = enc([q["question"] for q in cases], "query")
        tail_vectors = windows_bank[tail]
        tail_owner = owners[tail]

        def nearest(mode, alternate_slots=None):
            scores = queries @ docs.T
            if mode != "head":
                bank, slots = (tail_vectors, tail_owner) if mode == "tail" else (windows_bank, owners)
                if alternate_slots is not None:
                    slots = alternate_slots
                alternate = queries @ bank.T
                scores.scatter_reduce_(1, slots[None, :].expand_as(alternate), alternate, reduce="amax", include_self=True)
            return order(scores, cases)

        modes = ("head", "tail", "windows")
        sequence = Random(831543)
        results = {k: nearest(k) for k in modes}
        samples = {k: [] for k in modes}
        for _ in range(8):
            for k in sequence.sample(list(modes), len(modes)):
                torch.cuda.synchronize()
                start = perf_counter()
                rank = nearest(k)
                torch.cuda.synchronize()
                samples[k].append((perf_counter()-start)*1e3)
                assert rank == results[k]
        sham = {}
        if a.sham:
            for seed in SHAM_SEEDS:
                corrupted, changed = permuted_owners(owners.tolist(), lengths, seed)
                sham[str(seed)] = changed
                results[f"sham/{seed}"] = nearest("windows", torch.tensor(corrupted, device="cuda"))
    measured = []
    for i, row in enumerate(cases):
        lex = lexical["rows"][i]["ids"]
        values = {"bm25": lex[:LIMIT]}
        for mode, lists in results.items():
            values[f"dense_{mode}"] = lists[i][:LIMIT]
            values[f"hybrid_{mode}"] = rrf((lists[i], lex), (1., 1.))
        measured.append({"id": row["id"], "ranks": values})
    metadata = {"fixture_sha256": fixture_hash, "task": fixture["task"], "documents": len(docs), "window_tokens": cap,
                "device": torch.cuda.get_device_name(), "document_token_counts": lengths,
                "extra_windows": len(owners), "documents_with_tail": int(tail.sum()),
                "head_bank_bytes": docs.nbytes, "extra_bank_bytes": windows_bank.nbytes,
                "build_shard_seconds": [r["seconds"] for r in extra], "median_full_query_matrix_ms": {
                    k: median(v) for k, v in samples.items()}, "ms_replicates": samples}
    a.output.write_bytes(gzip.compress(json.dumps(metadata | {"rows": measured, "sham_moved_documents": sham},
                                                separators=(",", ":")).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("encode", "search"))
    for k in ("assets", "inputs", "output", "lexical"):
        p.add_argument("--" + k, type=Path, required=(k != "lexical"))
    for k in ("head", "extra"):
        p.add_argument("--" + k, type=Path, nargs=4)
    p.add_argument("--shard", type=int)
    p.add_argument("--sham", action="store_true")
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    fixture = json.loads((a.inputs/"fixture.json").read_bytes())
    revisions = json.loads((a.assets/"revisions.json").read_bytes())
    (encode if a.mode == "encode" else search)(a, fixture, revisions)


if __name__ == "__main__":
    main()
