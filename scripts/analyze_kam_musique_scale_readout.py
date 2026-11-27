#!/usr/bin/env python3
"""Compare four-paragraph readout for paired MuSiQue decomposer capacities."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_kam_musique_assignment import evidence
from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from audit_kam_natural_data import digest

SIZES = ("small", "base")
MODES = ("path", "greedy", "assignment", "oracle")
METRICS = ("complete_pool", "support", "em", "joint")


def score(cases, outcomes):
    clusters = shared_evidence_clusters(cases)
    counts, paired = {}, {}
    for i in range(3):
        flags = {name: {} for name in SIZES}
        for size in SIZES:
            for mode in MODES:
                rows = [outcomes[q["id"]][i][size][mode] for q in cases]
                flags[size][mode] = {k: [r[k] for r in rows] for k in METRICS}
        counts[i] = {size: {m: {k: sum(v) for k, v in row.items()} for m, row in arms.items()}
                     for size, arms in flags.items()}
        paired[i] = {"base_vs_small": {m: {k: compare(flags["base"][m][k], flags["small"][m][k], clusters)
                                          for k in METRICS} for m in MODES},
                     "base_greedy_vs_path": {k: compare(flags["base"]["greedy"][k], flags["base"]["path"][k], clusters)
                                             for k in METRICS}}
    return {"questions": len(cases), "clusters": len(clusters), "counts": counts, "paired": paired}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "output"):
        p.add_argument("--" + option, required=True, type=Path)
    for option, n in (("fit", 6), ("affinity", 24), ("reader", 4), ("baseline", 4)):
        p.add_argument("--" + option, required=True, type=Path, nargs=n)
    p.add_argument("--reference-ranks", type=Path, nargs=24)
    a = p.parse_args()
    manifest = unpack(a.manifest)
    cases = manifest["rows"]
    encoded = {digest(file): unpack(file) for file in a.affinity}
    baseline = {digest(file): unpack(file) for file in a.baseline}
    fits = [json.loads(f.read_text()) for f in a.fit]
    seeds = sorted({q["seed"] for q in fits})
    assert len(seeds) == 3 and all(q["model_size"] in SIZES and q["steps"] == 480 for q in fits)
    expected = {(v["seed"], v["model_size"]): v["revision"] for v in fits}
    originals = {}
    for trial in map(unpack, a.reference_ranks or []):
        originals[trial["shard"], trial["weights_sha256"]] = trial
    outputs, comparisons = {}, Counter()
    reader = sorted(map(unpack, a.reader), key=lambda r: r["shard"])
    weights = None
    for shard, measured in enumerate(reader):
        old = baseline[measured["baseline_sha256"]]
        work = [encoded[k] for k in measured["learned_sha256"]]
        assert len(work) == 6 and measured["device"] == "NVIDIA L4"
        for panel in (old, measured, *work):
            assert (panel["shard"], panel["shards"]) == (shard, 4)
            assert all(panel[k] == manifest[k] for k in ("fixture_sha256", "corpus_sha256"))
        assert all(v["baseline_sha256"] == measured["baseline_sha256"] for v in work)
        tags = [m["weights_sha256"] for m in work]
        assert len(set(tags)) == 6
        if weights is None:
            weights = tags
        assert weights == tags
        for q, start, result, *pools in zip(cases[shard::4], old["rows"], measured["rows"],
                                           *(m["rows"] for m in work), strict=True):
            id_ = q["id"]
            assert id_ not in outputs and all(r["id"] == id_ for r in (start, result, *pools))
            arms = result["arms"]
            assert arms["hybrid_rank"]["evidence_ids"] == start["hybrid"][:4]
            truth = set(q["gold"])
            outcomes = [dict() for _ in range(3)]
            for j, pool in enumerate(pools):
                tag, model = tags[j], SIZES[j % 2]
                assert len(pool["pool"]) == len(set(pool["pool"])) == 128
                direct = arms[f"learned_path/{j}"]["evidence_ids"]
                choices, _ = evidence(pool, direct)
                assert arms[f"learned_greedy/{j}"]["evidence_ids"] == choices["greedy_subquery"]
                assert arms[f"learned_assignment/{j}"]["evidence_ids"] == choices["joint_assignment"]
                out = {}
                for mode in MODES:
                    pred = arms[f"learned_{mode}/{j}"]
                    ids = pred["evidence_ids"]
                    assert len(ids) == len(set(ids)) == 4 and set(ids) <= set(pool["pool"])
                    support = truth <= set(ids)
                    out[mode] = {"complete_pool": truth <= set(pool["pool"]), "support": support,
                                 "em": pred["em"], "joint": support and pred["em"]}
                outcomes[j//2][model] = out
            outputs[id_] = outcomes
        for pool in work:
            reference = originals.get((shard, pool["weights_sha256"]))
            if reference:
                for row, previous in zip(pool["rows"], reference["rows"], strict=True):
                    assert {k: v for k, v in row.items() if k != "subquery_affinity"} == previous
                    comparisons["matched_candidate_replays"] += 1
    panels = {"all": cases}
    panels |= {str(h): [q for q in cases if q["hops"] == h] for h in (2, 3, 4) if any(q["hops"] == h for q in cases)}
    assert len(cases) == len(outputs)
    payload = {"fixture_sha256": manifest["fixture_sha256"], "seeds_by_index": seeds,
               "fit_revisions": {str(seed): {m: expected[seed, m] for m in SIZES} for seed in seeds},
               "provenance": comparisons, "weights": weights, "scopes": {n: score(rows, outputs) for n, rows in panels.items()}}
    a.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
