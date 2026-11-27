#!/usr/bin/env python3
"""Summarize native and FP32 choice changes under different batch packing."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(payload):
    rows = payload["records"]
    assert len(rows) == 2048 and len({row["index"] for row in rows}) == 2048
    gold = np.asarray([row["gold"] for row in rows])
    reference = np.asarray([row["original_choice"] for row in rows])
    outputs = {}
    for kind in ("native_scores", "float32_scores"):
        full = np.asarray([row["production"][kind] for row in rows])
        repacked = np.asarray([row["repacked"][kind] for row in rows])
        assert full.shape == repacked.shape == (len(rows), 4)
        assert np.isfinite(full).all() and np.isfinite(repacked).all()
        a = full.argmax(-1)
        b = repacked.argmax(-1)
        changed = a != b
        margin = np.diff(np.sort(full, axis=1)[:, -2:], axis=1)[:, 0]
        outputs[kind] = {"n": len(rows), "production_matches_original": int((a == reference).sum()),
                         "repacked_matches_original": int((b == reference).sum()),
                         "packing_choice_changes": int(changed.sum()),
                         "production_correct": int((a == gold).sum()), "repacked_correct": int((b == gold).sum()),
                         "production_tied_max": int((margin == 0).sum()),
                         "packing_changes_with_tied_production_max": int((changed & (margin == 0)).sum()),
                         "median_production_margin_changed": float(np.median(margin[changed])) if changed.any() else None,
                         "median_production_margin_stable": float(np.median(margin[~changed])) if (~changed).any() else None}
    return {"metadata": payload["metadata"], "gpu": payload["gpu"], "comparisons": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {p.stem: summarize(json.loads(p.read_text())) for p in args.replays}
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
