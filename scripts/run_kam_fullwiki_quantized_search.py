#!/usr/bin/env python3
"""Benchmark INT8 full-index shortlisting with FP32 reranking on frozen questions."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from kam_int8_index import INT_QUERY_ROWS, int8_encode, integer_product
from run_kam_fullwiki_coarse import distance, read
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_natural_feasibility import Encoder

WIDTH = 8
DEPTH = 64
SHORTLIST = (96, 192, 256)


def exact_rescore(query, bank, shortlist):
    vectors = bank[shortlist]
    scores = torch.mv(vectors, query)
    return shortlist[scores.topk(DEPTH).indices]


def smoke(device, dim):
    rng = torch.Generator(device=device).manual_seed(429417)
    x = torch.randint(-127, 128, (INT_QUERY_ROWS, dim), generator=rng, device=device, dtype=torch.int8)
    y = torch.randint(-127, 128, (256, dim), generator=rng, device=device, dtype=torch.int8)
    answer = torch._int_mm(x, y.T)
    assert torch.equal(answer.float(), x.float() @ y.float().T)


@torch.inference_mode()
def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    cfg, manifest = read(a.config), read(a.inputs / "manifest.json")
    fixture = read(a.fixture)
    raw = a.fixture.read_bytes()
    raw = gzip.decompress(raw) if a.fixture.suffix == ".gz" else raw
    assert fixture["source_sha256"] == manifest["source_sha256"]
    data = fixture["rows"]["test"]
    assert len(data) == 1024 and len({v["id"] for v in data}) == len(data)
    data = data[a.shard::a.shards]
    mapper = FullWikiIndex(a.inputs, manifest, cfg)
    full = mapper.bank
    half = mapper.bank_half
    if half is None:
        half = full.half()
    smoke(full.device, full.shape[1])
    torch.cuda.synchronize()
    ready = perf_counter()
    compressed, scale = int8_encode(full)
    padded_mask = torch.zeros(len(compressed), dtype=torch.bool, device=full.device)
    padded_mask[:len(full)] = mapper.mask
    torch.cuda.synchronize()
    build_ms = 1000*(perf_counter()-ready)

    def search(query, mode, caps):
        if mode == "fp16_rescore":
            rank = distance(query, half).masked_fill_(~mapper.mask, -torch.inf)
        else:
            rank = integer_product(query, compressed, scale).masked_fill_(~padded_mask, -torch.inf)
        proposals = rank.topk(max(caps), dim=-1).indices
        return {str(width): [exact_rescore(q, full, p[:width]).tolist()
                             for q, p in zip(query, proposals, strict=True)] for width in caps}

    encoder = Encoder(a.assets / "encoder", 128)
    queries = encoder([r["question"] for r in data], "query")
    assert len(data) % WIDTH == 0
    for mode in ("fp16_rescore", "int8_rescore"):
        for _ in range(2):
            search(queries[:WIDTH], mode, (SHORTLIST[0],))
    rng = Random(879331 + a.shard)
    output = []
    for begin in rng.sample(list(range(0, len(data), WIDTH)), len(data) // WIDTH):
        chunk = data[begin:begin + WIDTH]
        q = queries[begin:begin + WIDTH]
        numbers, timed = {}, {}
        modes = rng.sample(("fp16_rescore", "int8_rescore"), 2)
        for mode in modes:
            torch.cuda.synchronize()
            first = perf_counter()
            got = search(q, mode, (SHORTLIST[0],))
            torch.cuda.synchronize()
            timed[mode] = 1000*(perf_counter() - first)
            if mode == "int8_rescore":
                got |= search(q, mode, SHORTLIST[1:])
            numbers[mode] = got
        output.append({"ids": [item["id"] for item in chunk], "timing_ms": timed, "ranks": numbers})
    return {"fixture_sha256": sha256(raw).hexdigest(), "index_sha256": digest(a.inputs / "lexical/completed.json"),
            "models": read(a.assets/"revisions.json"), "shard": a.shard, "shards": a.shards,
            "device": torch.cuda.get_device_name(), "candidate_docs": mapper.meta["indexed"],
            "resident_bytes": {"fp16": half.nbytes, "int8": compressed.nbytes + scale.nbytes},
            "int8_build_ms": build_ms, "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "assets", "config", "fixture", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    a.output.write_bytes(gzip.compress(json.dumps(run(a), separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
