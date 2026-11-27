#!/usr/bin/env python3
"""Compare sentence encoder precision with the same search, reranker, and reader."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analyze_kam_native_musique import compare
from analyze_kam_shared_serving import unpack

ARMS = ("fp32", "bf16", "fp16")
STAGES = ("sentence_encoding_ms", "remaining_rerank_ms", "reader_ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("fixture", "output"):
        parser.add_argument("--" + field, required=True, type=Path)
    for field in ("measurements", "reference"):
        parser.add_argument("--" + field, required=True, nargs=4, type=Path)
    a = parser.parse_args()
    raw = gzip.decompress(a.fixture.read_bytes())
    fx = sha256(raw).hexdigest()
    gold = {r["id"]: set(r["gold_ids"]) for r in json.loads(raw)["rows"]["test"]}
    experimental, prior = [sorted(map(unpack, files), key=lambda r: r["shard"]) for files in (a.measurements, a.reference)]
    times = {m: {k: [] for k in (*STAGES, "request_ms")} for m in ARMS}
    flags = {m: {k: [] for k in ("support", "answer", "joint")} for m in ARMS}
    same = {m: Counter() for m in ARMS[1:]}
    seen = set()
    for shard, (current, original) in enumerate(zip(experimental, prior, strict=True)):
        assert current["shard"] == original["shard"] == shard
        assert current["fixture_sha256"] == original["fixture_sha256"] == fx
        assert current["gpu"] == original["gpu"] == "NVIDIA L4"
        assert all(current[k] == original[k] for k in ("models", "index_sha256", "fitted_sha256", "corpus_profile_partition"))
        assert len(current["rows"]) == len(original["rows"]) == 48 and set(current["search_modes"]) == set(ARMS)
        for row, old in zip(current["rows"], original["rows"], strict=True):
            assert row["id"] == old["id"] and row["id"] not in seen
            seen.add(row["id"])
            target = gold[row["id"]]
            full = row["records"]["fp32"]
            assert full == old["records"]["unbucketed"]
            for m in ARMS:
                got = row["records"][m]
                a_union = got["arms"]["conditioned_union"]
                if m != "fp32":
                    assert all(got[k] == full[k] for k in ("initial_ids", "first", "corpus_hits", "sentence_counts"))
                    same[m].update({"all_fields": got == full, "ranker_and_reader": got["arms"] == full["arms"],
                                    "conditioned_pair": a_union["pair"] == full["arms"]["conditioned_union"]["pair"]})
                support = set(a_union["pair"]) == target
                answer = a_union["exact"]
                for k, v in zip(flags[m], (support, answer, support and answer), strict=True):
                    flags[m][k].append(v)
                for key in STAGES:
                    times[m][key].append(row["phases_ms"][m][key])
                times[m]["request_ms"].append(row["request_ms"][m])
    n = len(seen)
    assert n == 192
    draws = np.random.default_rng(735812).integers(0, n, size=(4000, n))
    latency = {}
    for arm in ARMS:
        stages = {}
        for k, values in times[arm].items():
            row = np.asarray(values)
            paired = row - times["fp32"][k]
            stages[k] = {"p50": float(np.median(row)), "p95": float(np.quantile(row, .95)),
                         "paired_median_ms": float(np.median(paired)),
                         "paired_bootstrap95_ms": np.quantile(np.median(paired[draws], axis=1), [.025, .975]).tolist()}
        latency[arm] = stages
    quality = {m: {k: sum(v) for k, v in outcomes.items()} for m, outcomes in flags.items()}
    contrasts = {m: {k: compare(v, flags["fp32"][k], [[i] for i in range(n)]) for k, v in flags[m].items()}
                 for m in ARMS[1:]}
    result = {"fixture_sha256": fx, "requests": n, "unchanged_records": same, "latency": latency,
              "quality": quality, "quality_contrast": contrasts}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
