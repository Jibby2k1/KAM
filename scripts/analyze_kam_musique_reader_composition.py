#!/usr/bin/env python3
"""Audit matched evidence and measure four-hop MuSiQue composition with the fixed reader."""

import argparse
import json
from pathlib import Path

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack


def summarize(cases, records):
    output = {}
    for arm in records[0]["answers"]:
        correct, joint, control = [], [], []
        for q, r in zip(cases, records, strict=True):
            found = bool(r["answers"][arm])
            has_all = set(q["gold"]) <= set(r["evidence"])
            correct.append(found)
            joint.append(found and has_all)
            control.append(bool(r["answers"]["original_question"]))
        intervals = compare(correct, control, shared_evidence_clusters(cases))
        output[arm] = {"answer_exact": sum(correct), "joint_exact": sum(joint), "vs_original_question": intervals}
    return {"questions": len(cases), "archived_96_tokens_exact": sum(r["archived_exact"] for r in records),
            "full_evidence": sum(set(q["gold"]) <= set(r["evidence"]) for q, r in zip(cases, records, strict=True)),
            "correct_intermediate_steps": list(map(sum, zip(*(r["prediction_exact"] for r in records)))),
            "final_gold_binding_overlaps_answer": sum(r["gold_binding_matches_final"] for r in records),
            "rotation_changed": sum(r["rotation_active"] for r in records), "policies": output}


def audit(fixture, measurements):
    assert {r["fixture_sha256"] for r in measurements} == {fixture["fixture_sha256"]}
    assert len(measurements) == 4 and {r["shard"] for r in measurements} == set(range(4))
    assert {r["device"] for r in measurements} == {"NVIDIA L4"} and len({r["reader_revision"] for r in measurements}) == 1
    expected = [q for q in fixture["rows"] if q["hops"] == 4]
    panels = {}
    for group in ("gold_paragraphs", "primary_selected"):
        gathered = {}
        for block in measurements:
            determined = [q for q in fixture["rows"][block["shard"]::block["shards"]] if q["hops"] == 4]
            for r, q in zip(block["panels"][group], determined, strict=True):
                assert r["id"] == q["id"] and r["id"] not in gathered
                assert len(r["evidence"]) == len(set(r["evidence"])) == 4
                assert r["audit"]["evidence_tokens"] <= 4 * block["max_body_cap"]
                assert r["audit"]["total_length"] <= block["max_tokens"] == 512
                if group == "gold_paragraphs":
                    assert r["evidence"] == sorted(q["gold"])
                gathered[r["id"]] = r
        rows = [gathered[q["id"]] for q in expected]
        no_overlap = [(q, row) for q, row in zip(expected, rows, strict=True) if not row["gold_binding_matches_final"]]
        panels[group] = summarize(expected, rows)
        panels[group]["no_gold_overlap"] = summarize(*map(list, zip(*no_overlap, strict=True)))
    return {"fixture_sha256": fixture["fixture_sha256"], "analysis": panels}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--measurements", type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    results = audit(unpack(a.fixture), list(map(unpack, a.measurements)))
    a.output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
