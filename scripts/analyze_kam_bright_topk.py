#!/usr/bin/env python3
"""Audit exact rank changes and quality lost when extra windows can only rerank."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analyze_kam_bright_transfer import DEPTH, group_queries, limits, metrics, resample


def load(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def measure(manifest, reference_path, ranks):
    assert all(ranks[k] == manifest[k] for k in ("fixture_sha256", "task", "documents"))
    assert ranks["reference_sha256"] == sha256(reference_path.read_bytes()).hexdigest()
    reference = load(reference_path)
    assert reference["fixture_sha256"] == manifest["fixture_sha256"]
    original = [r | {"gold_docs": r["gold"]} for r in manifest["rows"]]
    matched = 0
    coverage, values, window_pairs = {}, {}, {}
    for truth, ref, row in zip(original, reference["rows"], ranks["rows"], strict=True):
        assert truth["id"] == ref["id"] == row["id"]
        gold, excluded = set(truth["gold"]), set(truth["excluded"])
        assert not truth["unindexed_gold"] and gold
        matched += row["ranks"]["hybrid_windows"] == ref["ranks"]["hybrid_windows"]
        for arm, docs in (list(row["ranks"].items()) + [("sealed_l4_windows", ref["ranks"]["hybrid_windows"])]):
            assert len(docs) == len(set(docs)) == DEPTH and not excluded.intersection(docs)
            values.setdefault(arm, []).append(metrics(docs, gold, len(gold)))
            original_set = set(ref["ranks"]["hybrid_windows"])
            coverage.setdefault(arm, []).append(len(original_set.intersection(docs)))
        for policy, count in row["window_pairs"].items():
            assert 0 <= count <= ranks["extra_windows"]
            window_pairs.setdefault(policy, []).append(count)
    assert matched == ranks["archive_matches"] and len(original) == ranks["within_device_exact"]
    groups = group_queries(original)
    sample = resample(groups)
    fields = ("ndcg10", "recall64", "full_recovery64")
    summaries = {}
    current = np.asarray(values["hybrid_windows"])
    head = np.asarray(values["hybrid_head"])
    for policy, vals in values.items():
        vals = np.asarray(vals)
        summaries[policy] = {"mean": dict(zip(fields, vals.mean(0).tolist(), strict=True)),
                             "relative_to_global_window": {k: {"paired_mean": float(v.mean()),
                                                               "cluster_bootstrap95": limits(v, groups, sample)}
                                                           for k, v in zip(fields, (vals - current).T, strict=True)},
                             "relative_to_head": {k: {"paired_mean": float(v.mean()), "cluster_bootstrap95": limits(v, groups, sample)}
                                                  for k, v in zip(fields, (vals - head).T, strict=True)},
                             "mean_overlap_with_frozen_l4": float(np.mean(coverage[policy]))}
        if policy in window_pairs:
            summaries[policy]["mean_eligible_window_pairs"] = float(np.mean(window_pairs[policy]))
            summaries[policy]["fraction_of_global_window_pairs"] = float(np.mean(window_pairs[policy]) / ranks["extra_windows"])
    return {k: v for k, v in ranks.items() if k != "rows"} | {"quality": summaries}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "reference", "measurement", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = measure(load(a.manifest), a.reference, load(a.measurement))
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
