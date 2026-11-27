#!/usr/bin/env python3
"""Measure hard span-reader transfer through fixed MuSiQue binding chains."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

ARMS = ("frozen", "trained")
METRICS = ("complete_pool", "first_binding", "penultimate_binding", "last_subanswer", "joint")


def flags(cases, rows, name):
    counts = {k: [] for k in METRICS}
    for q in cases:
        value = rows[q["id"]][name]
        exact = value["answer_em"]
        support = set(q["gold"]) <= set(value["pool"])
        for k, v in zip(METRICS, (support, exact[0], exact[-2], exact[-1], support and exact[-1]), strict=True):
            counts[k].append(bool(v))
    return counts


def assess(cases, by_seed):
    groups = shared_evidence_clusters(cases)
    result = {}
    for seed, content in by_seed.items():
        raw, scores = {}, {}
        for model, rows in content.items():
            names = set(rows[cases[0]["id"]])
            raw[model] = {name: flags(cases, rows, name) for name in sorted(names)}
            scores[model] = {}
            for name in raw[model]:
                entry = raw[model][name]
                scores[model][name] = {"counts": {k: sum(seq) for k, seq in entry.items()},
                                       "vs_flant5": {k: compare(seq, raw[model]["top1"][k], groups)
                                                     for k, seq in entry.items()}}
        result[str(seed)] = {"seed": seed, "readers": scores, "trained_vs_frozen": {
            name: {k: compare(raw["trained"][name][k], frozen[k], groups) for k in METRICS}
            for name, frozen in raw["frozen"].items()}}
    return {"questions": len(cases), "clusters": len(groups), "seeds": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name, amount in (("fit", 6), ("baseline", 4), ("measurement", 24)):
        p.add_argument("--" + name, type=Path, nargs=amount, required=True)
    a = p.parse_args()
    fixture, data = list(map(unpack, (a.manifest, a.probe)))
    assert data["native_sha256"] == fixture["fixture_sha256"]
    assert fixture["source_sha256"] == data["source_sha256"]
    fit = [json.loads(p.read_text()) for p in a.fit]
    training = {r["weights_sha256"]: r for r in fit}
    assert len(training) == 6 and all(r["data_sha256"] == data["data_sha256"] for r in fit)
    assert all(r["device"] == "NVIDIA L4" and r["steps"] == 720 and r["batch"] == 32 for r in fit)
    seeds = sorted({r["seed"] for r in fit})
    assert len(seeds) == 3
    for s in seeds:
        a_, b_ = sorted((r for r in fit if r["seed"] == s), key=lambda r: r["arm"])
        assert (a_["arm"], b_["arm"]) == ARMS
        assert all(a_[k] == b_[k] for k in ("initial_head_sha256", "sample_sha256", "fit", "monitor_records", "assets"))
    prior = {r["shard"]: r for r in map(unpack, a.baseline)}
    observed = {i: {m: {} for m in ARMS} for i in seeds}
    matches = Counter()
    for run in map(unpack, a.measurement):
        fit = training[run["span_checkpoint_sha256"]]
        key, model = fit["seed"], fit["arm"]
        shard = run["shard"]
        base = prior[shard]
        assert run["shards"] == base["shards"] == 4 and run["device"] == "NVIDIA L4"
        verify = ("fixture_sha256", "corpus_sha256", "models", "templates_sha256", "scoring_labels_sha256", "baseline_sha256")
        assert all(base[k] == run[k] for k in verify)
        for q, row, source in zip(fixture["rows"][shard::4], run["rows"], base["rows"], strict=True):
            id_, hops = q["id"], q["hops"]
            assert id_ == row["id"] == source["id"] and id_ not in observed[key][model]
            old = source["arms"]["top1"]
            assert old == row["arms"]["top1"]
            matches[model] += 1
            first_use = {"span_first": 0, "span_confident_fallback": 0}
            if hops > 2:
                first_use |= {"span_penultimate": hops-2, "span_all_intermediate": 0}
            assert set(row["arms"]) == {"top1", *first_use}
            for policy, index in first_use.items():
                trial = row["arms"][policy]
                for reading in ("target_ranks", "reader_context_ids", "gold_in_reader"):
                    assert len(trial[reading]) == hops and old[reading][:index+1] == trial[reading][:index+1]
                assert old["answer_em"][:index] == trial["answer_em"][:index]
                assert trial["pool"][:64] == old["pool"][:64] and len(set(trial["pool"])) == len(trial["pool"]) == 128
            observed[key][model][id_] = row["arms"]
    cases = fixture["rows"]
    assert len(cases) == 576 and all(x.keys() == {q["id"] for q in cases} for s in observed.values() for x in s.values())
    summary = {str(h): assess([q for q in cases if q["hops"] == h], observed) for h in COHORTS}
    result = {"fixture_sha256": fixture["fixture_sha256"], "training_data_sha256": data["data_sha256"],
              "resolved_reference_rows": matches, "fits": list(training.values()), "scopes": summary}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
