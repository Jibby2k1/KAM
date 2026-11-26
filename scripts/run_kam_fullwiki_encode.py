#!/usr/bin/env python3
"""Encode one reproducible contiguous whole-corpus slice."""

import argparse
import json
from pathlib import Path
from time import perf_counter

# Load the PyTorch module runtime before Arrow loads its C++ dependencies.
import torch
import pyarrow.parquet as pq

from run_kam_natural_feasibility import Encoder


def run(a):
    begin = perf_counter()
    manifest = json.loads(a.manifest.read_text())
    sizes = manifest["row_groups"]
    group, part = a.shard % len(sizes), a.shard // len(sizes)
    assert 0 <= part < a.parts
    parquet = pq.ParquetFile(a.corpus)
    assert [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)] == sizes
    lo = part * sizes[group] // a.parts
    hi = (part + 1) * sizes[group] // a.parts
    offset = sum(sizes[:group]) + lo
    selected = parquet.read_row_group(group, columns=["title", "text"]).slice(lo, hi - lo).to_pylist()
    texts = [str(row["title"] or "") + ". " + str(row["text"] or "") for row in selected]
    del selected
    order = sorted(range(len(texts)), key=lambda i: min(192, len(texts[i].split())))
    transformed = [texts[i] for i in order]
    encoder = Encoder(a.assets / "encoder", a.batch)
    start = perf_counter()
    values = encoder(transformed, "passage").cpu()
    inverse = torch.tensor(order).argsort()
    vectors = values.index_select(0, inverse).contiguous()
    assert vectors.dtype == torch.float32 and torch.isfinite(vectors).all()
    elapsed = perf_counter() - start
    output = a.output / f"vectors_{a.shard:02d}.pt"
    staging = output.with_suffix(".tmp")
    torch.save({"offset": offset, "vectors": vectors, "source_sha256": manifest["source_sha256"],
                "device": torch.cuda.get_device_name(), "shard": a.shard}, staging)
    staging.replace(output)
    print(json.dumps({"shard": a.shard, "offset": offset, "docs": len(texts), "torch": torch.__version__,
                      "device": torch.cuda.get_device_name(), "encoding_seconds": elapsed,
                      "process_seconds": perf_counter() - begin,
                      "docs_per_second": len(texts) / elapsed}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "corpus", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--parts", type=int, default=6)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--shard", type=int, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
