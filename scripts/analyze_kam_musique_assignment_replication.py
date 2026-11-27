#!/usr/bin/env python3
"""Audit joint and greedy paragraph assignments on a fresh MuSiQue split."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_musique_assignment import evidence
from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

SEEDS = (106331, 106339, 106367)
METHODS = ("hybrid_rank", "gold_only", *(f"learned_{mode}/{i}"
           for i in range(3) for mode in ("path", "assignment", "greedy", "oracle")))
METRICS = ("support", "em", "joint")


def outcomes(cases, read):
    flags = {m: {k: [] for k in METRICS} for m in METHODS}
    for case in cases:
        gold = set(case["gold"])
        arms = read[case["id"]]
        for m in METHODS:
            row = arms[m]
            support = gold <= set(row["evidence_ids"])
            for key, v in zip(METRICS, (support, row["em"], support and row["em"]), strict=True):
                flags[m][key].append(v)
    group = shared_evidence_clusters(cases)
    counts = {m: {k: sum(v) for k, v in metrics.items()} for m, metrics in flags.items()}
    result = {}
    for i in range(3):
        joint = f"learned_assignment/{i}"
        result[str(SEEDS[i])] = {control: {k: compare(flags[joint][k], flags[control][k], group) for k in METRICS}
                                 for control in (f"learned_path/{i}", f"learned_greedy/{i}", "hybrid_rank")}
    return {"questions": len(cases), "shared_title_clusters": len(group), "counts": counts, "paired": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    for name, n in (("reader", 4), ("search", 4), ("affinity", 12)):
        p.add_argument("--" + name, required=True, type=Path, nargs=n)
    a = p.parse_args()
    fixture = unpack(a.manifest)
    cases = fixture["rows"]
    key = ("fixture_sha256", "corpus_sha256")
    expected = {k: fixture[k] for k in key}

    def index(paths):
        return {sha256(p.read_bytes()).hexdigest(): unpack(p) for p in paths}

    search, affinity = index(a.search), index(a.affinity)
    finals = sorted(map(unpack, a.reader), key=lambda q: q["shard"])
    assert len(search) == len(finals) == 4 and len(affinity) == 12
    assert [r["shard"] for r in finals] == list(range(4))
    observed, equal, notes = {}, Counter(), {}
    for output in finals:
        shard = output["shard"]
        baseline = search[output["baseline_sha256"]]
        paths = [affinity[h] for h in output["learned_sha256"]]
        for detail in (output, baseline, *paths):
            assert all(detail[k] == expected[k] for k in key) and (detail["shard"], detail["shards"]) == (shard, 4)
        assert output["device"] == "NVIDIA L4"
        assert all(detail["baseline_sha256"] == output["baseline_sha256"] for detail in paths)
        assert output["label_sha256"] == finals[0]["label_sha256"] and output["reader_revision"] == finals[0]["reader_revision"]
        weights = [p["weights_sha256"] for p in paths]
        if not notes:
            notes = {"weights": weights, "plan_hashes": [p["generation_sha256"] for p in paths]}
        assert weights == notes["weights"] and [p["generation_sha256"] for p in paths] == notes["plan_hashes"]
        for case, first, answer, *learned in zip(cases[shard::4], baseline["rows"], output["rows"],
                                                 *(p["rows"] for p in paths), strict=True):
            assert case["id"] == first["id"] == answer["id"] and all(x["id"] == case["id"] for x in learned)
            arms = answer["arms"]
            assert arms["hybrid_rank"]["evidence_ids"] == first["hybrid"][:4]
            for i, variant in enumerate(learned):
                direct = arms[f"learned_path/{i}"]["evidence_ids"]
                picks, changed = evidence(variant, direct)
                joint = arms[f"learned_assignment/{i}"]
                greedy = arms[f"learned_greedy/{i}"]
                assert joint["evidence_ids"] == picks["joint_assignment"]
                assert greedy["evidence_ids"] == picks["greedy_subquery"]
                equal[f"{i}/assignment_changed_from_greedy"] += changed
                equal[f"{i}/same_order"] += joint["evidence_ids"] == greedy["evidence_ids"]
                equal[f"{i}/same_set"] += set(joint["evidence_ids"]) == set(greedy["evidence_ids"])
            observed[case["id"]] = arms
    assert len(cases) == len(observed)
    panels = {str(hops): [q for q in cases if q["hops"] == hops] for hops in (2, 3, 4)}
    data = expected | notes | {"primary_hops": fixture["primary_hops"], "agreement": equal,
                               "scopes": {k: outcomes(s, observed) for k, s in panels.items() if s}}
    a.output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
