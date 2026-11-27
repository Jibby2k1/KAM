#!/usr/bin/env python3
"""Measure paired full-pipeline throughput and output stability across batches."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analyze_kam_native_musique import compare
from analyze_kam_shared_serving import unpack

WIDTHS = ("1", "2", "4", "8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in ("batch", "reference"):
        p.add_argument("--" + name, type=Path, nargs=4, required=True)
    p.add_argument("--confirm-fp16", action="store_true")
    a = p.parse_args()
    data = gzip.decompress(a.fixture.read_bytes())
    fx = sha256(data).hexdigest()
    truth = {q["id"]: set(q["gold_ids"]) for q in json.loads(data)["rows"]["test"]}
    shards, reference = [sorted(map(unpack, inputs), key=lambda r: r["shard"]) for inputs in (a.batch, a.reference)]
    trial = {name: int(name) for name in WIDTHS}
    baseline = "1"
    if a.confirm_fp16:
        trial = {f"{p}/{w}": w for p in ("fp32", "fp16") for w in (1, 8)}
        baseline = "fp32/1"
    groups = {w: [] for w in trial}
    spans = {w: [] for w in trial}
    outcomes = {w: {"support": [], "answer": [], "joint": []} for w in trial}
    same = {w: Counter() for w in trial}
    encoded = {w: 0 for w in trial}
    ids = set()
    for i, (probe, old) in enumerate(zip(shards, reference, strict=True)):
        assert probe["shard"] == old["shard"] == i and probe["fixture_sha256"] == old["fixture_sha256"] == fx
        assert all(probe[k] == old[k] for k in ("models", "index_sha256", "fitted_sha256", "corpus_profile_partition", "gpu"))
        assert probe["gpu"] == "NVIDIA L4" and probe["shards"] == old["shards"] == 4
        assert len(probe["rows"]) == (25 if a.confirm_fp16 else 6) and len(old["rows"]) == 48
        expected = {q["id"]: q["records"]["shortlist_text"] for q in old["rows"]}
        for group in probe["rows"]:
            labels = group["ids"]
            assert len(labels) == 8 and not (ids & set(labels))
            assert not a.confirm_fp16 or not set(labels).intersection(expected)
            ids.update(labels)
            assert set(map(str, group["order"])) == set(trial) == set(group["batches"])
            original = group["batches"][baseline]["rows"]
            for w, record in group["batches"].items():
                timing, rows = record["timing"], record["rows"]
                assert len(timing) == 8 // trial[w] and all(t["size"] == trial[w] for t in timing)
                assert [r["id"] for r in rows] == labels
                encoded[w] += record["sentence_counts"]["encoded"]
                groups[w].append(sum(t["request_ms"] for t in timing))
                spans[w].extend(t["request_ms"] for t in timing)
                for case, row, base in zip(labels, rows, original, strict=True):
                    if not a.confirm_fp16:
                        control = {k: v for k, v in expected[case].items() if k not in ("sentence_normalization", "sentence_counts")}
                        same[w]["archived_single_record"] += row == control
                    same[w].update({"all_fields": row == base,
                                    "candidate_lists": all(row[k] == base[k] for k in ("initial_ids", "corpus_hits", "first")),
                                    "all_ranking_and_answer_records": row["arms"] == base["arms"],
                                    "conditioned_pair": row["arms"]["conditioned_union"]["pair"] == base["arms"]["conditioned_union"]["pair"]})
                    out = row["arms"]["conditioned_union"]
                    support = set(out["pair"]) == truth[case]
                    for key, bit in zip(outcomes[w], (support, out["exact"], support and out["exact"]), strict=True):
                        outcomes[w][key].append(bit)
    n = len(ids)
    assert n == (800 if a.confirm_fp16 else 192)
    assert a.confirm_fp16 or same[baseline]["archived_single_record"] == n
    base = np.asarray(groups[baseline])
    samples = np.random.default_rng(93114).integers(0, len(base), size=(4000, len(base)))
    performance = {}
    for w, rows in groups.items():
        durations = np.asarray(rows)
        paired = base / durations
        performance[w] = {"requests_per_second": n / (durations.sum() / 1000),
                          "group_p50_ms": float(np.median(durations)),
                          "batch_p50_ms": float(np.median(spans[w])),
                          "batch_p95_ms": float(np.quantile(spans[w], .95)),
                          "median_group_speedup": float(np.median(paired)),
                          "median_group_speedup_95": np.quantile(np.median(paired[samples], axis=1), [.025, .975]).tolist()}
    counts = {a: {k: sum(v) for k, v in flags.items()} for a, flags in outcomes.items()}
    contrasts = {a: {k: compare(v, outcomes[baseline][k], [[i] for i in range(n)]) for k, v in flags.items()}
                 for a, flags in outcomes.items() if a != baseline}
    out = {"fixture_sha256": fx, "requests": n, "groups": len(base), "performance": performance, "same": same,
           "encoded_sentences": encoded, "outcomes": counts, "quality_vs_single_fp32": contrasts}
    a.output.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
