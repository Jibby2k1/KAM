#!/usr/bin/env python3
"""Measure frozen 2Wiki retrieval and equal-depth conditioned expansion."""

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_conditioned_corpus_search import top_corpus
from run_kam_fullwiki_native_compare import FullWikiIndex, reciprocal_rank_fusion
from run_kam_hotpot_independent import read
from run_kam_natural_feasibility import Encoder, candidate_texts


@torch.inference_mode()
def run(a):
    start = perf_counter()
    cfg, fixture, manifest = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json"))
    assert fixture["rows"] and manifest["source_sha256"] == fixture["corpus"]["source_sha256"]
    index = FullWikiIndex(a.inputs, manifest, cfg)
    assert all(index.meta[k] == v for k, v in fixture["corpus"].items())
    encoder = Encoder(a.assets / "encoder", 96)
    cases = fixture["rows"][a.shard::a.shards]
    rows = []
    for i in range(0, len(cases), 16):
        sample = cases[i:i+16]
        texts = [r["question"] for r in sample]
        query = encoder(texts, "query")
        lexical = [index.lexical(txt, 256) for txt in texts]
        dense, _ = top_corpus(index, query, [-1] * len(sample), depth=256)
        anchor = [d[0] for d in dense]
        arguments = [candidate_texts(index.candidates({"question": q}, [hop]))[0]
                     for q, hop in zip(texts, anchor, strict=True)]
        conditioned = encoder([f"{q} First retrieved document: {text}" for q, text in zip(texts, arguments, strict=True)],
                              "query")
        hits, _ = top_corpus(index, (query + conditioned) * 0.5, anchor, depth=128)
        for case, bm25, semantic, extra in zip(sample, lexical, dense, hits, strict=True):
            original = semantic[:64]
            originals = set(original)
            novel = [p for p in extra if p not in originals][:64]
            assert len(novel) == len(set(novel)) == 64
            hybrid = reciprocal_rank_fusion(bm25, semantic, 128, index.rank_constant)
            rows.append({"id": case["id"], "bm25": bm25[:128], "dense": semantic[:128],
                         "hybrid": hybrid, "conditioned": original + novel})
    return {"fixture_sha256": digest(a.fixture), "config_sha256": digest(a.config), "corpus": fixture["corpus"],
            "encoder": read(a.assets / "revisions.json"), "device": torch.cuda.get_device_name(), "shard": a.shard,
            "shards": a.shards, "elapsed_seconds": perf_counter() - start, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "config", "fixture", "inputs", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    a.output.write_bytes(gzip.compress(json.dumps(run(a), separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
