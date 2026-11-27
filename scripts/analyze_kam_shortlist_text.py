#!/usr/bin/env python3
"""Verify shortlist-only text normalization during paired live requests."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles

import numpy as np

from analyze_kam_shared_serving import unpack

MODES = ("all_text", "shortlist_text")


def evaluate(rows):
    normalize = {name: [] for name in MODES}
    latencies = {name: [] for name in MODES}
    stage = {name: [] for name in MODES}
    n_equal = 0
    for request in rows:
        assert set(request["order"]) == set(MODES)
        recorded = request["records"]
        summaries = []
        for name in MODES:
            item = recorded[name]
            before = {k: v for k, v in item.items() if k != "sentence_normalization"}
            summaries.append(before)
            data = item["sentence_normalization"]
            assert data["normalized_documents"] <= data["candidate_documents"]
            normalize[name].append(data["normalized_documents"])
            latencies[name].append(request["request_ms"][name])
            stage[name].append(request["phases_ms"][name]["remaining_rerank_ms"])
        n_equal += summaries[0] == summaries[1]
        assert recorded["all_text"]["sentence_normalization"]["candidate_documents"] == normalize["all_text"][-1]
    sampling = np.random.default_rng(291481).integers(0, len(rows), size=(4000, len(rows)))
    differences = {label: np.asarray(data["shortlist_text"]) - data["all_text"]
                   for label, data in (("request", latencies), ("rerank", stage))}
    return {"requests": len(rows), "same_rank_and_reader_records": n_equal,
            "normalized_docs_p50": {m: median(times) for m, times in normalize.items()},
            "normalized_docs_total": {m: sum(times) for m, times in normalize.items()},
            "request_p50_ms": {m: median(times) for m, times in latencies.items()},
            "request_p95_ms": {m: quantiles(times, n=100, method="inclusive")[94] for m, times in latencies.items()},
            "rerank_p50_ms": {m: median(times) for m, times in stage.items()},
            "paired_changes": {key: {"p50_ms": float(np.median(d)),
                                     "bootstrap95_ms": np.quantile(np.median(d[sampling], axis=1), [0.025, 0.975]).tolist()}
                               for key, d in differences.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--measurements", required=True, nargs=4, type=Path)
    a = p.parse_args()
    raw = gzip.decompress(a.fixture.read_bytes())
    fx = sha256(raw).hexdigest()
    ids = {q["id"] for q in json.loads(raw)["rows"]["test"]}
    measurements = sorted(map(unpack, a.measurements), key=lambda c: c["shard"])
    read = measurements[0]
    assert [m["shard"] for m in measurements] == list(range(4))
    for c in measurements:
        assert c["fixture_sha256"] == fx and c["gpu"] == "NVIDIA L4" and c["shards"] == 4 and len(c["rows"]) == 48
        assert set(c["search_modes"]) == set(MODES)
        assert all(c[k] == read[k] for k in ("corpus_profile_partition", "search_modes", "models", "index_sha256", "fitted_sha256"))
    rows = [p for c in measurements for p in c["rows"]]
    outcomes = {q["id"] for q in rows}
    assert len(outcomes) == len(rows) == 192 and outcomes <= ids
    summary = {"fixture_sha256": fx, "settings": read["search_modes"], **evaluate(rows)}
    a.output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
