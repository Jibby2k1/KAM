#!/usr/bin/env python3
"""Separate observed HotpotQA expansion failures from policy-selection headroom."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_dense_hybrid_anchor import audit_method, digest, unpack

METHODS = ("dense", "hybrid")
POLICIES = ("broad", "broad_anchor")
STAGES = ("coverage", "shortlist", "support", "joint_em")
PRIMARY = "hybrid_broad_anchor"


def load(study):
    fixture_raw = gzip.decompress((study / "fixture.json.gz").read_bytes())
    questions = json.loads(fixture_raw)["rows"]["test"]
    fixture = {q["id"]: set(q["gold_ids"]) for q in questions}
    assert len(fixture) == len(questions) == 1024

    method_labels, candidates, refs = {}, {}, {}
    for method in METHODS:
        method_labels[method], metadata = audit_method(study, method, fixture, sha256(fixture_raw).hexdigest())
        assert metadata["broad_anchor_gpus"] == ["NVIDIA B200"] * 4
        assert metadata["repeated_broad_matches"] == len(fixture)
        pool_path = study / f"pools-{method}.json.gz"
        replay_path = study / f"{method}-broad-replay.json.gz"
        pools = unpack(pool_path)["rows"]
        logs = {r["id"]: r for s in unpack(replay_path) for r in s["rows"]}
        candidates[method] = {i: set(pools[i]) | set(logs[i]["corpus_hits"]["conditioned_union"]) for i in fixture}
        refs[method] = {"pool_sha256": digest(pool_path), "b200_replay_sha256": digest(replay_path),
                        "quality_config_sha256": metadata["config_sha256"], "index_sha256": metadata["index_sha256"]}
    assert refs["dense"]["quality_config_sha256"] == refs["hybrid"]["quality_config_sha256"]
    assert refs["dense"]["index_sha256"] == refs["hybrid"]["index_sha256"]
    rows = []
    for id_ in sorted(fixture):
        arms = {f"{m}_{p}": method_labels[m][id_][p] for m in METHODS for p in POLICIES}
        gold = fixture[id_]
        original_missing = method_labels["hybrid"][id_]["missing"]
        expanded = [candidates[m][id_] for m in METHODS]
        assert all(arms[f"{m}_broad_anchor"]["coverage"] == (gold <= c) for m, c in zip(METHODS, expanded, strict=True))
        rows.append({"arms": arms, "hybrid_original_missing": original_missing,
                     "merged_expansion_coverage": gold <= expanded[0] | expanded[1],
                     "merged_expansion_size": len(expanded[0] | expanded[1])})
    return rows, refs


def summarize(rows):
    n = len(rows)
    measures = {a: {s: sum(r["arms"][a][s] for r in rows) for s in STAGES}
                for a in (f"{m}_{p}" for m in METHODS for p in POLICIES)}
    pathway = {x: {name: n - result["coverage"] if i == 0 else result[STAGES[i - 1]] - result[stage]
                   for i, (name, stage) in enumerate(zip(
                       ("missing_from_pool", "removed_in_shortlist", "lost_in_final_rerank", "reader_inexact"), STAGES,
                       strict=True))}
               for x, result in measures.items()}
    for a in measures:
        assert sum(pathway[a].values()) + measures[a]["joint_em"] == n
    oracle_groups = {"choose_between_expansions": ("dense_broad_anchor", "hybrid_broad_anchor"),
                     "choose_between_all_four": tuple(measures)}
    oracles = {k: {s: sum(any(r["arms"][a][s] for a in arms) for r in rows) for s in STAGES}
               for k, arms in oracle_groups.items()}
    ceilings = {s: oracles["choose_between_all_four"][s] - measures[PRIMARY][s] for s in STAGES}
    contributions = Counter()
    joint = [r for r in rows if not r["arms"][PRIMARY]["joint_em"]]
    for row in joint:
        for arm, value in row["arms"].items():
            if value["joint_em"]:
                contributions[arm] += 1
    return {"questions": n, "measures": measures, "first_failure": pathway, "policy_choice_oracle": oracles,
            "policy_choice_headroom_over_hybrid_expansion": ceilings,
            "joint_rescues_by_arm_among_hybrid_expansion_misses": dict(sorted(contributions.items())),
            "merged_candidate_coverage": sum(r["merged_expansion_coverage"] for r in rows),
            "merged_candidate_size_mean": round(sum(r["merged_expansion_size"] for r in rows) / n, 6) if n else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows, provenance = load(a.study)
    grouped = {"all": rows}
    for missing in range(3):
        grouped[f"hybrid_original_missing_{missing}"] = [r for r in rows if r["hybrid_original_missing"] == missing]
    result = {"refs": provenance, "groups": {k: summarize(v) for k, v in grouped.items()}}
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
