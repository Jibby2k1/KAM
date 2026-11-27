#!/usr/bin/env python3
"""Validate sparse beam work counts and summarize matched CPU scoring timings."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

from analyze_kam_shared_serving import metrics, unpack

VARIANTS = ("neutral", "title")
SPARSE = ("unique", "mask")


def examine(profile, trace_path, static_path):
    assert profile["replay_sha256"] == sha256(trace_path.read_bytes()).hexdigest()
    assert profile["static_sha256"] == sha256(static_path.read_bytes()).hexdigest()
    trace = unpack(trace_path)
    rows = profile["rows"]
    n = len(rows)
    assert n == len(trace) == 1200
    for r, ref in zip(rows, trace, strict=True):
        assert (r["id"], r["source_id"]) == (ref["id"], ref["source_id"])
        for v in VARIANTS:
            c = r["cases"][v]
            assert c["row_counts"]["dense"] == ref["all_unordered_pairs"]
            assert all(c["row_counts"][m] == ref["distinct_pairs_within_beam"]["4"] for m in SPARSE)
            assert set(c["runtime_microseconds"]) == {"dense", *SPARSE}
            assert all(len(times) == profile["repeats"] > 0 and min(times) > 0
                       for times in c["runtime_microseconds"].values())

    def compare(variant, kind):
        paired = []
        for r in rows:
            values = r["cases"][variant]["runtime_microseconds"]
            paired.append({"serving_timing_ms": {"reference": median(values["dense"]) / 1000,
                                                 "shared": median(values[kind]) / 1000}})
        t = metrics(paired)
        return {"dense": t.pop("reference"), kind: t.pop("shared"), **t}

    return {"host": profile["host"], "questions": n, "repeats": profile["repeats"],
            "mean_pairs": {k: round(mean(r["cases"]["neutral"]["row_counts"][k] for r in rows), 6)
                           for k in ("dense", *SPARSE)},
            "comparisons": {v: {k: compare(v, k) for k in SPARSE} for v in VARIANTS}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for x in ("profile", "reference", "static", "output"):
        p.add_argument("--" + x, type=Path, required=True)
    a = p.parse_args()
    result = examine(unpack(a.profile), a.reference, a.static)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
