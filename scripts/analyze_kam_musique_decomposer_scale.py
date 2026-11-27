#!/usr/bin/env python3
"""Assess small versus base MuSiQue decomposers on the same frozen paragraph bank."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_kam_musique_atomic_bindings import authored, refs
from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from audit_kam_natural_data import digest

SIZES = ("small", "base")
METRICS = ("complete_support", "valid_plan", "correct_depth", "author_dependency_graph")


def summarize(cases, outputs):
    result = {}
    clusters = shared_evidence_clusters(cases)
    for seed, mapping in outputs.items():
        rows = {size: [mapping[size][q["id"]] for q in cases] for size in SIZES}
        result[str(seed)] = {"counts": {size: dict(sum((Counter(r) for r in data), Counter())) for size, data in rows.items()},
                             "base_vs_small": {k: compare([r[k] for r in rows["base"]],
                                                           [r[k] for r in rows["small"]], clusters) for k in METRICS}}
    return {"questions": len(cases), "clusters": len(clusters), "seeds": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "source", "training_manifest", "output"):
        p.add_argument("--" + option.replace("_", "-"), type=Path, required=True)
    for option, n in (("fit", 6), ("plans", 6), ("ranks", 24), ("baseline", 4)):
        p.add_argument("--" + option, type=Path, nargs=n, required=True)
    a = p.parse_args()
    fixture, training = map(unpack, (a.manifest, a.training_manifest))
    gold_ref = authored(a.source, fixture)
    cases = fixture["rows"]
    assert len(cases) == 576 and Counter(q["hops"] for q in cases) == COHORTS
    rank_panels, fit_by_seed = {}, {}
    for raw in map(unpack, a.ranks):
        rank_panels.setdefault(raw["generation_sha256"], []).append(raw)
    for p in a.fit:
        trained = json.loads(p.read_text())
        assert trained["device"] == "NVIDIA B200" and not trained["binding_tokens"]
        assert trained["fixture_sha256"] == fixture["fixture_sha256"] == training["fixture_sha256"]
        assert trained["seed_per_hop"] == 6 and trained["steps"] == 480
        fit_by_seed.setdefault(trained["seed"], {})[trained["model_size"]] = trained
    assert len(fit_by_seed) == 3
    assert len({r["data_sha256"] for by_size in fit_by_seed.values() for r in by_size.values()}) == 1
    baseline = {digest(file): unpack(file) for file in a.baseline}
    outputs = {}
    for i, (seed, runs) in enumerate(sorted(fit_by_seed.items())):
        outputs[seed] = {}
        assert set(runs) == set(SIZES)
        for size in SIZES:
            plan_file, = [p for p in a.plans if p.name.endswith(f"-{i}-{size}.json.gz")]
            plan = unpack(plan_file)
            assert plan["revision"] == runs[size]["revision"] and plan["fixture_sha256"] == fixture["fixture_sha256"]
            assert plan["device"] == "NVIDIA B200" and (plan["shard"], plan["shards"]) == (0, 1)
            ranks = sorted(rank_panels[digest(plan_file)], key=lambda r: r["shard"])
            assert len(ranks) == 4
            found = {}
            for shard, rank in enumerate(ranks):
                base = baseline[rank["baseline_sha256"]]
                assert (rank["shard"], base["shard"], rank["shards"]) == (shard, shard, 4)
                assert rank["weights_sha256"] == plan["weights_sha256"] and rank["device"] == "NVIDIA L4"
                assert rank["corpus_sha256"] == base["corpus_sha256"] == fixture["corpus_sha256"]
                for q, query, r, b in zip(cases[shard::4], plan["rows"][shard::4], rank["rows"], base["rows"], strict=True):
                    id_ = q["id"]
                    assert id_ == query["id"] == r["id"] == b["id"]
                    assert len(set(r["pool"])) == len(r["pool"]) == 128
                    assert bool(r["fallback"]) != query["valid"]
                    if r["fallback"]:
                        assert r["pool"] == b["hybrid"]
                    graph = refs(query["steps"])
                    assert r["predicted_steps"] == len(graph)
                    found[id_] = dict(complete_support=set(q["gold"]) <= set(r["pool"]), valid_plan=query["valid"],
                                      correct_depth=len(graph) == q["hops"], author_dependency_graph=graph == gold_ref[id_],
                                      hybrid_complete_support=set(q["gold"]) <= set(b["hybrid"]))
            outputs[seed][size] = found
    scopes = {"all": cases} | {str(h): [q for q in cases if q["hops"] == h] for h in COHORTS}
    result = {"fixture_sha256": fixture["fixture_sha256"], "fits": fit_by_seed,
              "scopes": {name: summarize(items, outputs) for name, items in scopes.items()}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
