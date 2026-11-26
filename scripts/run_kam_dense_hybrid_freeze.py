#!/usr/bin/env python3
"""Seal paired depth-64 dense and hybrid pools from one shared corpus scan."""

import argparse
import gzip
import json
from pathlib import Path

import torch

from audit_kam_natural_data import digest
from run_kam_fullwiki_native_compare import FullWikiIndex, reciprocal_rank_fusion
from run_kam_hotpot_independent import read
from run_kam_natural_feasibility import Encoder


@torch.inference_mode()
def run(a):
    manifest = read(a.inputs / "manifest.json")
    fixture, cfg = read(a.fixture), read(a.config)
    assert fixture["source_sha256"] == manifest["source_sha256"]
    cases = fixture["rows"]["test"]
    assert len(cases) == 1024 and len({q["id"] for q in cases}) == len(cases)
    cases = cases[a.shard::a.shards]
    encoder = Encoder(a.assets / "encoder", 128)
    queries = encoder([q["question"] for q in cases], "query")
    index = FullWikiIndex(a.inputs, manifest, cfg)
    rows = {"dense": [], "hybrid": []}
    for q, vector in zip(cases, queries.split(1), strict=True):
        dense = index.search("dense", q["question"], vector, 128)
        lexical = index.lexical(q["question"], 128)
        hybrid = reciprocal_rank_fusion(lexical, dense, 64, index.rank_constant)
        rows["dense"].append({"id": q["id"], "row_ids": dense[:64]})
        rows["hybrid"].append({"id": q["id"], "row_ids": hybrid})
    meta = {"fixture_sha256": digest(a.fixture), "index_sha256": digest(a.inputs / "lexical" / "completed.json"),
            "config_sha256": digest(a.config), "shard": a.shard, "shards": a.shards,
            "gpu": torch.cuda.get_device_name(), "rrf_rank_constant": index.rank_constant,
            "component_depth": 128, "pool_depth": 64}
    for method, result in rows.items():
        payload = json.dumps({**meta, "method": method, "rows": result}, separators=(",", ":"), allow_nan=False)
        (a.output / f"freeze-{method}-{a.shard}.json.gz").write_bytes(gzip.compress(payload.encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "config", "fixture", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    run(a)


if __name__ == "__main__":
    main()
