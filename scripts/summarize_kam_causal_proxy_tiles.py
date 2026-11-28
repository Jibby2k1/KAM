#!/usr/bin/env python3
"""Verify route parity and replay proxy-only, attention, and prefill timings."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json

PROTOCOL = "kam-causal-proxy-tiling-v1"
ROUTES = {"rectangular", "causal-tiles", "causal-local"}
SHAPES = {f"{batch}x{width}" for batch in (1,4) for width in (4096,8192)}


def read(path):
    return json.loads(path.read_text())


def timing(rows, repetitions):
    result = {}
    for name,r in rows.items():
        wall, cuda = map(np.array, (r["wall_ms"],r["cuda_ms"]))
        assert len(wall) == len(cuda) == repetitions and all(np.isfinite(a).all() and min(a)>0 for a in (wall,cuda))
        result[name] = {"cuda_p50_ms": float(np.median(cuda)), "wall_p50_ms": float(np.median(wall)),
                        "wall_p95_ms": float(np.quantile(wall,.95)), "peak_bytes": r["incremental_allocated_bytes"]}
    return result


def summarize(kernel, model, fixture, prior):
    k, m = map(read, (kernel, model))
    assert k["protocol"] == m["protocol"] == PROTOCOL and k["gpu"] == m["gpu"]
    assert m["manifest_sha256"] == digest(fixture) and m["seed"] == SEEDS[0]
    with gzip.open(prior, "rt") as stream:
        original = json.loads(json.load(stream)[f"transfer-{SEEDS[0]}-flash"])
    assert m["checkpoint_sha256"] == original["checkpoint_sha256"]
    assert set(k["timings"]) == set(m["timings"]) == SHAPES
    assert set(k["parity"]) == SHAPES | {"1x103","2x1024","1x2749"}
    assert set(m["parity"]) == {"4096","8192"}
    for p in k["parity"].values():
        assert set(p) == ROUTES-{"rectangular"}
        assert all(r["whole_route_equal"] and r["output_equal"] and r["changed_route_elements"]==0
                   and r["max_abs_output_delta"]==0 for r in p.values())
    for p in m["parity"].values():
        assert len(p)==4 and all(r["exact"] and r["values"]==16 and r["max_abs_delta"]==0 for r in p.values())
    measurements = {}
    for shape in sorted(SHAPES):
        assert all(set(v) == ROUTES for v in k["timings"][shape].values())
        local = {tier: timing(values, 30) for tier, values in k["timings"][shape].items()}
        complete = timing(m["timings"][shape],15)
        relative = {}
        for a,b in complete.items():
            if ":" in a:
                original = complete[a.split(":")[-1]]
                relative[a] = 1-b["wall_p50_ms"]/original["wall_p50_ms"]
        measurements[shape] = local | {"full_model": complete, "fraction_wall_saved_vs_same_policy": relative}
    return {"protocol": PROTOCOL, "gpu": k["gpu"], "torch": k["torch"], "manifest_sha256": m["manifest_sha256"],
            "kernel_sha256":digest(kernel), "model_sha256":digest(model), "synthetic_parity":k["parity"],
            "full_model_parity":m["parity"], "measurements": measurements}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("kernel","model","fixture","prior","output"):
        cli.add_argument(name,type=Path)
    a=cli.parse_args()
    write_json(a.output,summarize(a.kernel,a.model,a.fixture,a.prior))
