#!/usr/bin/env python3
"""Compare calendar and relative session dates with matched evidence and prompt text."""

import argparse
import json
from pathlib import Path

from analyze_kam_longmemeval_timeline_order import eligible
from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from run_kam_longmemeval_m_reader import CLOCK_EVIDENCE, RELATIVE_CLOCKS

SIZES = ("base", "large")
FLAGS = ("answer", "joint", "abstained")


def hits(rows, amp):
    evidence = [r["marked_support"] for r in rows]
    scores = [r["readers"][amp] for r in rows]
    answer = [r["em"] for r in scores]
    return {"answer": answer, "joint": [a and e for a, e in zip(answer, evidence, strict=True)],
            "abstained": [r["abstained"] for r in scores]}


def assess(cases, evaluated):
    groups = shared_evidence_clusters([{"gold": q["gold_ids"] or [q["id"]]} for q in cases])
    arms = {}
    for name in CLOCK_EVIDENCE:
        scopes = {}
        for size in SIZES:
            by_clock = {clock: hits([evaluated[q["id"]][f"{name}/{clock}"] for q in cases], size)
                        for clock in RELATIVE_CLOCKS}
            scopes[size] = {k: {"counts": {c: sum(rows[c]) for c in FLAGS},
                                "paired_vs_calendar": {c: compare(rows[c], by_clock["calendar"][c], groups) for c in FLAGS}}
                            for k, rows in by_clock.items()}
        arms[name] = scopes
    return {"questions": len(cases), "shared_evidence_clusters": len(groups), "arms": arms}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("measurement", "reference"):
        p.add_argument("--" + name, type=Path, required=True, nargs=4)
    a = p.parse_args()
    archive = unpack(a.manifest)
    cases = archive["rows"]
    archived = sorted(map(unpack, a.reference), key=lambda t: t["shard"])
    runs = sorted(map(unpack, a.measurement), key=lambda t: t["shard"])
    matched = {}
    time_by_reader = []
    for i, (run, ref) in enumerate(zip(runs, archived, strict=True)):
        assert run["shard"] == ref["shard"] == i and run["shards"] == ref["shards"] == 4
        assert run["device"] == ref["device"] == "NVIDIA L4"
        assert all(run[tag] == ref[tag] for tag in ("fixture_sha256", "reference_sha256", "labels_sha256", "weights"))
        assert run["fixture_sha256"] == archive["fixture_sha256"]
        time_by_reader.append(run["prediction_seconds"])
        for q, actual, before in zip(cases[i::4], run["rows"], ref["rows"], strict=True):
            assert q["id"] == actual["id"] == before["id"] and q["id"] not in matched
            arms = actual["arms"]
            assert set(arms) == {f"{x}/{y}" for x in CLOCK_EVIDENCE for y in RELATIVE_CLOCKS}
            for evidence in CLOCK_EVIDENCE:
                initial = before["arms"][f"{evidence}_rank_dated"]
                baseline = arms[f"{evidence}/calendar"]
                assert baseline["turns"] == initial["turns"] == initial["prompt_order"]
                assert baseline["marked_support"] == initial["marked_support"]
                for tag in RELATIVE_CLOCKS:
                    saw = arms[f"{evidence}/{tag}"]
                    assert saw["dated"] and saw["clock_encoding"] == tag and saw["prompt_tokens"] <= 1024
                    assert all(saw[k] == baseline[k] for k in ("turns", "prompt_order", "body_sha256", "clipped_turns",
                                                               "retained_turn_tokens", "marked_support", "shifted_turns"))
            matched[q["id"]] = arms
    assert len(matched) == len(cases) == 500
    answerable = [q for q in cases if not q["id"].endswith("_abs")]
    temporal = [q for q in answerable if q["kind"] == "temporal-reasoning"]
    scopes = {"answerable": answerable, "temporal": temporal,
              "temporal_known_as_of": [q for q in temporal if eligible(q)],
              "temporal_later_evidence": [q for q in temporal if not eligible(q)],
              "non_temporal": [q for q in answerable if q["kind"] != "temporal-reasoning"],
              "abstention": [q for q in cases if q["id"].endswith("_abs")]}
    result = {"fixture_sha256": archive["fixture_sha256"], "reader_weights": runs[0]["weights"],
              "matched_question_ids": len(matched), "prediction_seconds": time_by_reader,
              "scopes": {name: assess(panel, matched) for name, panel in scopes.items()}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
