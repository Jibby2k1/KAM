#!/usr/bin/env python3
"""Audit sentence inference bucketing and profile its paired latency."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analyze_kam_shared_serving import unpack

ARMS = ("unbucketed", "bucketed")
STAGES = ("sentence_encoding_ms", "rerank_through_sentence_ms", "remaining_rerank_ms", "reader_ms")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("fixture", "output"):
        p.add_argument("--" + field, required=True, type=Path)
    for field in ("measurement", "reference"):
        p.add_argument("--" + field, required=True, nargs=4, type=Path)
    a = p.parse_args()
    raw = gzip.decompress(a.fixture.read_bytes())
    fx = sha256(raw).hexdigest()
    truth = {q["id"]: set(q["gold_ids"]) for q in json.loads(raw)["rows"]["test"]}
    experiment = sorted(map(unpack, a.measurement), key=lambda c: c["shard"])
    reference = sorted(map(unpack, a.reference), key=lambda c: c["shard"])
    assert [r["shard"] for r in experiment] == list(range(4))
    latency = {name: {key: [] for key in (*STAGES, "request_ms")} for name in ARMS}
    outcomes = {m: Counter() for m in ARMS}
    seen, equivalent = set(), Counter()
    for current, past in zip(experiment, reference, strict=True):
        assert current["shard"] == past["shard"] and current["fixture_sha256"] == past["fixture_sha256"] == fx
        for key in ("models", "index_sha256", "fitted_sha256", "corpus_profile_partition", "gpu"):
            assert current[key] == past[key]
        assert current["gpu"] == "NVIDIA L4"
        assert set(current["search_modes"]) == set(ARMS) and len(current["rows"]) == len(past["rows"]) == 48
        for row, old in zip(current["rows"], past["rows"], strict=True):
            case = row["id"]
            assert old["id"] == case and case in truth and case not in seen
            seen.add(case)
            ranks = {arm: row["records"][arm] for arm in ARMS}
            anchor = {k: v for k, v in old["records"]["shortlist_text"].items() if k != "sentence_normalization"}
            assert ranks["unbucketed"] == anchor
            equivalent.update({"all_records": ranks["unbucketed"] == ranks["bucketed"],
                               "candidate_pool": all(ranks["unbucketed"][k] == ranks["bucketed"][k]
                                                     for k in ("initial_ids", "first", "corpus_hits"))})
            for arm in ARMS:
                phase = row["phases_ms"][arm]
                for key in STAGES:
                    latency[arm][key].append(phase[key])
                latency[arm]["request_ms"].append(row["request_ms"][arm])
                out = ranks[arm]["arms"]["conditioned_union"]
                support = set(out["pair"]) == truth[case]
                outcomes[arm].update({"support": support, "answer_em": out["exact"],
                                      "joint_em": support and out["exact"]})
    n = len(seen)
    assert n == 192 and equivalent["candidate_pool"] == n
    rng = np.random.default_rng(35168).integers(0, n, size=(4000, n))
    data = {}
    for key in latency["unbucketed"]:
        t = np.asarray([latency[arm][key] for arm in ARMS])
        delta = t[1] - t[0]
        data[key] = {"p50": dict(zip(ARMS, np.median(t, axis=1).tolist(), strict=True)),
                     "p95": dict(zip(ARMS, np.quantile(t, .95, axis=1).tolist(), strict=True)),
                     "paired_p50": float(np.median(delta)),
                     "paired_bootstrap95": np.quantile(np.median(delta[rng], axis=1), [.025, .975]).tolist()}
    result = {"fixture_sha256": fx, "requests": n, "outcomes": outcomes, "identical": equivalent, "latency": data}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
