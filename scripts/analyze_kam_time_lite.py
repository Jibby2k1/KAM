#!/usr/bin/env python3
"""Audit matched passages and compare answer accuracy on frozen TIME-Lite choices."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from analyze_kam_bright_transfer import group_queries, limits, resample

CONTENT_FIXED = ("dense_date_masked", "dense_chronological", "dense_dates_rotated")


def read(file):
    return json.loads(gzip.decompress(file.read_bytes()) if file.suffix == ".gz" else file.read_bytes())


def summarize(truth, reader):
    assert truth["fixture_sha256"] == reader["fixture_sha256"]
    input_groups = truth["source_groups"]
    cases = truth["rows"]
    group = group_queries([q | {"gold_docs": input_groups[q["context"]]} for q in cases])
    sample = resample(group)
    per_policy = {}
    for q, row in zip(cases, reader["rows"], strict=True):
        assert q["id"] == row["id"]
        assert set(reader["reader_seconds"]) == set(row["logits"]) == set(row["prompt"])
        prompt = row["prompt"]
        reference = prompt["dense"]["rotations"]
        for arm in set(CONTENT_FIXED).intersection(prompt):
            assert prompt[arm]["selected"] == prompt["dense"]["selected"] == row["dense"]
            assert prompt[arm]["rotations"] == reference
        assert all(len(arm["selected"]) <= reader["views"] for arm in prompt.values())
        assert all(m["total_length"] <= reader["input_tokens_max"]
                   for arm in prompt.values() for m in arm["rotations"])
        for arm, logits in row["logits"].items():
            n = q["options"]
            assert all(len(v) == n for v in logits) and len(logits) == n
            data = np.asarray(logits)
            assert np.isfinite(data).all()
            rotations = data.argmax(axis=1)
            answers = (int(data.mean(axis=0).argmax()), int(rotations[0]),
                       float(np.mean(rotations == q["gold"])), int(np.unique(rotations).size == 1))
            fields = per_policy.setdefault(arm, {})
            fields[q["id"]] = [int(answers[0] == q["gold"]), int(answers[1] == q["gold"]),
                               answers[2], answers[3], answers[0]]
    metrics = {}
    subset = {task: np.array([i for i, q in enumerate(cases) if q["family"] == task])
              for task in sorted({r["family"] for r in cases})}
    dense = np.asarray([per_policy["dense"][q["id"]][0] for q in cases])
    for arm, values in sorted(per_policy.items()):
        all_values = np.asarray([values[q["id"]] for q in cases])
        correct = all_values[:, 0]
        delta = correct - dense
        metrics[arm] = {"n": len(cases), "correct": int(sum(correct)),
                        "unrotated_correct": int(sum(all_values[:, 1])), "mean_accuracy_per_rotation": float(all_values[:, 2].mean()),
                        "rotation_consistent": int(sum(all_values[:, 3])), "vs_dense": {
                            "change": float(delta.mean()), "source_cluster_bootstrap95": limits(delta, group, sample)},
                        "per_family": {task: {"n": int(len(ids)), "correct": int(sum(correct[ids]))} for task, ids in subset.items()},
                        "reader_seconds": reader["reader_seconds"][arm],
                        "packed_body_tokens": sum(sum(x["body_tokens"] for x in r["prompt"][arm]["rotations"]) for r in reader["rows"])}
        meta = [r["prompt"][arm] for r in reader["rows"]]
        if arm == "dense_dates_rotated":
            changed = np.asarray([p["rotated_date_changes"] > 0 for p in meta])
            metrics[arm]["changed_panels"] = {"n": int(changed.sum()), "better": int(sum(delta[changed] > 0)),
                                               "worse": int(sum(delta[changed] < 0))}
        if arm == "dense_chronological":
            changes = sum(p["order"] != list(range(len(p["selected"]))) for p in meta)
            metrics[arm]["changed_order"] = int(changes)
    return {"fixture": reader["fixture_sha256"], "device": reader["device"], "reader_model": reader["model"],
            "queries": len(cases), "source_contexts": truth["contexts"], "source_clusters": len(group),
            "largest_shared_source_cluster": max(map(len, group)), "reader_token_budget": reader["input_tokens_max"],
            "max_windows": reader["views"], "time_encode_rank_seconds": reader["encode_rank_seconds"],
            "metrics": metrics}, per_policy, group


def analyze(manifest, readers):
    domains = manifest["domains"]
    assert len(readers) == len({(r["domain"], r["size"]) for r in readers}) == 2 * len(domains)
    assert len({r["device"] for r in readers}) == 1
    summaries = {}
    for name, domain in domains.items():
        a, b = sorted([r for r in readers if r["domain"] == name], key=lambda r: r["size"])
        assert a["size"] == "base" and b["size"] == "large"
        assert a["encoder_revisions"] == b["encoder_revisions"]
        for x, y in zip(a["rows"], b["rows"], strict=True):
            assert x["id"] == y["id"] and x["dense"] == y["dense"] and x["prompt"] == y["prompt"]
        base, one, groups = summarize(domain, a)
        large, two, same = summarize(domain, b)
        assert groups == same
        sample = resample(groups)
        compare = {}
        for arm in one:
            delta = np.asarray([two[arm][q["id"]][0] - one[arm][q["id"]][0] for q in domain["rows"]])
            compare[arm] = {"gain": float(delta.mean()), "source_cluster_bootstrap95": limits(delta, groups, sample),
                            "reader_time_ratio": large["metrics"][arm]["reader_seconds"] / base["metrics"][arm]["reader_seconds"]}
        summaries[name] = {"base": base, "large": large, "large_vs_base": compare}
    return {"source_revision": manifest["source_revision"], "domains": summaries}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--measurements", nargs=6, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(analyze(read(a.manifest), list(map(read, a.measurements))), sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
