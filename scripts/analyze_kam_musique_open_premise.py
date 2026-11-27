#!/usr/bin/env python3
"""Reconcile open first-hop results with the fixed endpoint quality panel."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_musique_endpoint_pairs import groups
from analyze_kam_native_musique import compare
from analyze_kam_shared_serving import unpack


def audit(records, references):
    previous = {unpack(p)["shard"]: p for p in references}
    live = list(map(unpack, records))
    assert len(live) == 4 and {r["shard"] for r in live} == set(previous) == set(range(4))
    assert {r["device"] for r in live} == {"NVIDIA L4"}
    cases = []
    provenance = live[0]
    for result in live:
        for field in ("fixture_sha256", "corpus_sha256", "source_sha256", "models"):
            assert result[field] == provenance[field]
        ref = previous[result["shard"]]
        original = unpack(ref)
        assert result["prior_sha256"] == sha256(ref.read_bytes()).hexdigest()
        assert original["model_revisions"] == result["models"]
        assert original["fixture_sha256"] == result["fixture_sha256"]
        earlier = [x for x in original["rows"] if x["hops"] == 2 and not x["reverse"]]
        for a, b in zip(earlier, result["rows"], strict=True):
            assert all(a[k] == b[k] for k in ("pair", "question", "family"))
            assert b["archived_joint_oracles"] == {k: v["support_exact"] and v["exact"] for k, v in a["routed"]["large"].items()}
            for method, v in b["open_routes"].items():
                expected = b["gold_earlier"]
                pool = v["prefix_candidates"]
                assert len(pool) == len(set(pool)) == 8
                assert v["premise_rank"] == (1 + pool.index(expected) if expected in pool else None)
                assert v["premise_correct"] == (v["selected_premise"] == expected)
                assert v["selected_premise"] in pool[:int(method.split("/")[1])]
            cases.append(b)
    assert len(cases) == len({q["question"] for q in cases}) == 168
    families = groups(cases, "family")
    discovered = {}
    question_only = [x["archived_joint_oracles"]["query_route"] for x in cases]
    true_premise = [x["archived_joint_oracles"]["true_prefix_route_endpoint_only"] for x in cases]
    for policy in cases[0]["open_routes"]:
        rows = [r["open_routes"][policy] for r in cases]
        joint = [x["answer_exact"] and x["endpoint_correct"] for x in rows]
        chosen = [x["premise_correct"] for x in rows]
        discovered[policy] = {"premise_in_top_eight": sum(x["premise_rank"] is not None for x in rows),
                              "premise_in_top_four": sum(x["premise_rank"] is not None and x["premise_rank"] <= 4 for x in rows),
                              "premise_chosen": sum(chosen), "endpoint_selected": sum(x["endpoint_correct"] for x in rows),
                              "answer_exact": sum(x["answer_exact"] for x in rows), "endpoint_joint": sum(joint),
                              "full_chain_joint": sum(x and y for x, y in zip(chosen, joint, strict=True)),
                              "foil_answer": sum(x["foil_answer"] for x in rows),
                              "vs_query_first": compare(joint, question_only, families),
                              "vs_true_earlier": compare(joint, true_premise, families)}
    prior = {k: sum(x["archived_joint_oracles"][k] for x in cases) for k in cases[0]["archived_joint_oracles"]}
    return {"questions": len(cases), "families": len(families), "archived_controls": prior, "routes": discovered,
            "fixture_sha256": provenance["fixture_sha256"], "corpus_sha256": provenance["corpus_sha256"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, nargs=4, required=True)
    p.add_argument("--references", type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(audit(args.measurements, args.references), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
