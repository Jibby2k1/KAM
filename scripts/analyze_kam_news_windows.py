#!/usr/bin/env python3
"""Audit news fact visibility and exact answers under fixed article IDs."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from prepare_kam_news_windows import KINDS
from run_kam_news_windows import ARMS

METRICS = ("all_facts_visible", "em", "joint")


def summarize(cases, observed):
    clusters = shared_evidence_clusters([{"gold": q["article_ids"]} for q in cases])
    flags = {a: {k: [] for k in METRICS} for a in ARMS}
    counts = {}
    for arm in ARMS:
        rows = [observed[q["id"]][arm] for q in cases]
        for row in rows:
            cover = row["visible_facts"] == row["fact_count"]
            for key, bit in zip(METRICS, (cover, bool(row["em"]), cover and bool(row["em"])), strict=True):
                flags[arm][key].append(bit)
        counts[arm] = {k: sum(v) for k, v in flags[arm].items()} | {
            "mean_f1": mean(r["f1"] for r in rows), "median_body_tokens": median(r["context_body_tokens"] for r in rows)}
    differences = (("sentence_center", "stride"), ("two_centers", "stride"), ("two_centers", "sentence_center"),
                   ("gold_text", "no_evidence"), ("sentence_center", "no_evidence"))
    audits = {f"{a}_vs_{b}": {k: compare(flags[a][k], flags[b][k], clusters) for k in METRICS} for a, b in differences}
    return {"questions": len(cases), "clusters": len(clusters), "largest_cluster": max(map(len, clusters)),
            "counts": counts, "paired": audits}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--measurements", type=Path, required=True, nargs=4)
    a = p.parse_args()
    sealed = unpack(a.manifest)
    cases = sealed["rows"]
    runs = sorted(map(unpack, a.measurements), key=lambda r: r["shard"])
    assert [x["shard"] for x in runs] == list(range(4))
    output = {}
    for shard, data in enumerate(runs):
        assert data["device"] == "NVIDIA L4" and data["shards"] == 4
        assert all(data[k] == sealed[k] for k in ("fixture_sha256", "corpus_sha256"))
        assert all(data[k] == runs[0][k] for k in ("reader_weights_sha256", "reader_revision"))
        for case, row in zip(cases[shard::4], data["rows"], strict=True):
            id_ = row["id"]
            assert case["id"] == id_ and set(row["arms"]) == set(ARMS)
            assert len(case["article_ids"]) == 2
            for arm in row["arms"].values():
                assert arm["fact_count"] >= 2 and 0 <= arm["visible_facts"] <= arm["fact_count"]
                assert 0 < arm["input_tokens"] <= 512 and 0 <= arm["context_body_tokens"] <= 384
            output[id_] = row["arms"]
    assert len(cases) == len(output) == 384
    scopes = {"all": cases} | {k: [q for q in cases if q["kind"] == k] for k in KINDS}
    result = {"fixture_sha256": sealed["fixture_sha256"], "scopes": {k: summarize(group, output) for k, group in scopes.items()}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
