#!/usr/bin/env python3
"""Summarize frozen cross-domain pair selection on audited per-question results."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_kam_fullwiki_support_links import paired
from audit_kam_natural_data import digest

ARMS = ("learned_single_shot", "learned_sequential", "one_pass_pair",
        "matched_plain", "context_only", "support_link")
RANK_ARMS = ARMS[2:]


def joint(row, arm):
    out = row["arms"][arm]
    return bool(out["both_support"] and out["em"])


def stats(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    ranking = [r for r in rows if r["ranking"]]
    oracle = [r["arms"]["oracle"] for r in rows if "oracle" in r["arms"]]
    return {
        "n": n,
        "both_in_pool": len(oracle),
        "oracle_answer": sum(r["em"] for r in oracle),
        "arms": {arm: {
            "both_support": sum(r["arms"][arm]["both_support"] for r in rows),
            "answer_em": sum(r["arms"][arm]["em"] for r in rows),
            "joint_em": sum(joint(r, arm) for r in rows),
            "wrong_linked": sum(r["arms"][arm]["selected_linked"] and not r["arms"][arm]["both_support"]
                                for r in rows),
            "both_exact_evidence_spans": sum(r["arms"][arm].get("both_evidence_spans", False) for r in rows),
        } for arm in ARMS},
        "linked_contests": len(ranking),
        "linked_top": {arm: sum(r["ranking"][arm] for r in ranking) for arm in RANK_ARMS},
        "context_joint_differences": {
            arm: paired([joint(r, "context_only") - joint(r, arm) for r in rows])
            for arm in ("one_pass_pair", "learned_sequential", "matched_plain", "support_link")
        },
    }


def summarize(shards, fixture):
    contents = json.loads(fixture.read_text())
    assert all(run["dataset"] == contents["dataset"] and run["fixture_sha256"] == digest(fixture)
               for run in shards)
    for key in ("baseline_sha256", "heads_sha256"):
        assert len({run[key] for run in shards}) == 1
    rows = []
    for i, run in enumerate(shards):
        expected = contents["rows"][i::len(shards)]
        assert [q["id"] for q in expected] == [q["id"] for q in run["rows"]]
        for row, case in zip(run["rows"], expected, strict=True):
            assert row["depth"] >= 3 and row["kind"] == case["kind"]
            gold = set(row["gold"])
            assert all(0 <= g < row["depth"] for g in gold)
            if "gold" in case:
                assert gold == set(case["gold"])
            for arm in ARMS:
                r = row["arms"][arm]
                assert len(set(r["pair"])) == 2 and all(0 <= g < row["depth"] for g in r["pair"])
                assert r["both_support"] == (set(r["pair"]) == gold)
                if r["both_support"]:
                    assert row["gold_linked"] == r["selected_linked"]
            rows.append(row)
    return {"dataset": contents["dataset"], "source_sha256": contents["source_sha256"],
            "recruitment": contents["counts"], "candidate_depth_counts": dict(Counter(r["depth"] for r in rows)),
            "groups": {
                "all": stats(rows),
                "linked": stats([r for r in rows if r["gold_linked"] is True]),
                "unlinked": stats([r for r in rows if r["gold_linked"] is False]),
                "missing": stats([r for r in rows if r["gold_linked"] is None]),
            },
            "subtypes": {kind: stats([r for r in rows if r["kind"] == kind])
                         for kind in sorted({r["kind"] for r in rows})},
            "both_spans_visible": sum(len(r.get("visible_gold", [])) == 2 for r in rows),
            "elapsed_seconds": [run["elapsed_seconds"] for run in shards],
            "gpu": [run["gpu"] for run in shards]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("shards", nargs="+", type=Path)
    a = p.parse_args()
    results = summarize([json.loads(p.read_text()) for p in a.shards], a.fixture)
    a.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
