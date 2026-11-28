#!/usr/bin/env python3
"""Verify CUDA graph reuse and summarize synchronized timings by shape."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import digest
from kam_math_kl_protocol import write_json

ARMS = ("flash", "protected", "tail-256")
SHAPES = ("1x4096", "1x8192", "2x4096", "2x8192", "4x4096", "4x8192")


def summarize(path, manifest):
    profile = json.loads(path.read_text())
    assert profile["manifest_sha256"] == digest(manifest) and set(profile["shapes"]) == set(SHAPES)
    result = {}
    for shape in SHAPES:
        batch, width = map(int, shape.split("x"))
        row = profile["shapes"][shape]
        assert set(row["capture"]) == set(ARMS)
        assert set(row["timings"]) == {f"{style}/{arm}" for style in ("eager", "graph") for arm in ARMS}
        by_arm = {}
        for arm in ARMS:
            proof = row["capture"][arm]
            assert proof["bitwise_parity"] and proof["max_abs_logit_delta"] == proof["updated_input_max_abs_delta"] == 0
            measured = {}
            for style in ("eager", "graph"):
                trial = row["timings"][f"{style}/{arm}"]
                wall, cuda = map(np.asarray, (trial["wall_ms"], trial["cuda_ms"]))
                assert len(wall) == len(cuda) == 12 and all(np.isfinite(a).all() and min(a)>0 for a in (wall, cuda))
                p50 = float(np.median(wall))
                measured[style] = {"wall_p50_ms": p50, "wall_p95_ms": float(np.quantile(wall, .95)),
                                   "cuda_p50_ms": float(np.median(cuda)), "tokens_per_second": batch*width*1000/p50,
                                   "incremental_allocated_bytes": trial["incremental_allocated_bytes"]}
            by_arm[arm] = {**measured, "capture": proof,
                "graph_minus_eager_fraction": measured["graph"]["wall_p50_ms"]/measured["eager"]["wall_p50_ms"] - 1}
        result[shape] = by_arm
    return {"profile_sha256": digest(path), "gpu": profile["gpu"], "torch": profile["torch"],
            "manifest_sha256": profile["manifest_sha256"], "checkpoint_sha256": profile["checkpoint_sha256"], "shapes": result}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "manifest", "summary"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    write_json(a.summary, summarize(a.profile, a.manifest))
