#!/usr/bin/env python3
"""Audit scarce-window retrieval and executable reasoning on BABILong."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from analyze_kam_native_musique import compare
from analyze_kam_shared_serving import unpack
from audit_kam_babilong_replay import TASKS
from run_kam_babilong_retrieval import BUDGETS

METRICS = ("complete_witness", "exact", "joint")


def score(cases, rows, budget):
    flags = {}
    scope = rows[cases[0]["id"]]["budgets"][str(budget)]
    for method in scope:
        measurements = [rows[q["id"]]["budgets"][str(budget)][method] for q in cases]
        flags[method] = {k: [bool(m[k] and q["evidence"]) if k == "complete_witness" else bool(m[k])
                             for q, m in zip(cases, measurements, strict=True)] for k in METRICS}
    groups = [[i] for i in range(len(cases))]
    return {"questions": len(cases), "with_witness": sum(bool(q["evidence"]) for q in cases),
            "counts": {m: {k: sum(v) for k, v in metrics.items()} for m, metrics in flags.items()},
            "paired_vs_dense": {m: {k: compare(v, flags["dense"][k], groups) for k, v in metrics.items()}
                                for m, metrics in flags.items() if m != "dense"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "output"):
        p.add_argument("--" + option, required=True, type=Path)
    p.add_argument("--measurements", type=Path, nargs="+", required=True)
    p.add_argument("--b200-control", type=Path)
    a = p.parse_args()
    source = unpack(a.manifest)
    findings, observations, fixture_rows = {}, {}, {}
    for run in map(unpack, a.measurements):
        task, length = run["task"], run["length"]
        assert task in TASKS and length in ("16k", "64k")
        assert run["fixture_sha256"] == source["fixture_sha256"]
        assert run["device"] == "NVIDIA L4"
        cases = [q for q in source["rows"] if q["task"] == task and q["length"] == length]
        assert [q["id"] for q in cases] == [r["id"] for r in run["rows"]]
        key = f"{task}/{length}"
        assert key not in findings
        rows, cardinality = {}, Counter()
        for q, room in zip(cases, run["rows"], strict=True):
            rows[q["id"]] = room
            total, modeled = room["windows"], room["event_windows"]
            assert 0 < modeled <= total
            assert set(room["budgets"]) == {str(i) for i in BUDGETS}
            for kind, arms in room["budgets"].items():
                cap = min(int(kind), total)
                names = {"lexical", "dense", "grammar_dense"} | ({"sequential"} if task in ("qa2", "qa3") else set())
                assert set(arms) == names
                for method, found in arms.items():
                    selected = found["windows"]
                    expected = min(cap, modeled) if method == "grammar_dense" else cap
                    assert len(selected) == len(set(selected)) == expected and all(0 <= i < total for i in selected)
                    assert found["events"] >= found["witness_spans_recovered"]
                    assert found["complete_witness"] == (found["witness_spans_recovered"] == len(q["evidence"]))
                    assert found["joint"] == bool(found["complete_witness"] and found["exact"] and q["evidence"])
            cardinality.update(windows=total, event_windows=modeled)
        scopes = {"all": cases}
        if task in ("qa7", "qa8"):
            scopes["nonempty"] = [case for case in cases if case["gold"]]
            scopes["empty"] = [case for case in cases if not case["gold"]]
        findings[key] = {"device": run["device"], "n": len(cases), "windows": cardinality,
                         "median_windows": median(r["windows"] for r in rows.values()),
                         "median_event_windows": median(r["event_windows"] for r in rows.values()),
                         "gpu_panel_seconds": run["elapsed_seconds"],
                         "scopes": {k: {str(i): score(v, rows, i) for i in BUDGETS} for k, v in scopes.items()}}
        observations[key], fixture_rows[key] = rows, cases
    paired_lengths = {}
    for task in TASKS:
        left, right = (f"{task}/{length}" for length in ("16k", "64k"))
        if left not in observations or right not in observations:
            continue
        base, added = [observations[key] for key in (left, right)]
        cases = fixture_rows[left]
        assert list(base) == list(added)
        def outcome(rows, q, arm, cap, prop):
            value = rows[q["id"]]["budgets"][str(cap)][arm][prop]
            return bool(value and q["evidence"]) if prop == "complete_witness" else value

        groups = [[i] for i in range(len(cases))]
        paired_lengths[task] = {
            str(cap): {m: {k: compare([outcome(added, q, m, cap, k) for q in cases],
                                     [outcome(base, q, m, cap, k) for q in cases], groups) for k in METRICS}
                       for m in base[cases[0]["id"]]["budgets"][str(cap)]} for cap in BUDGETS}
    result = {"fixture_sha256": source["fixture_sha256"], "panels": findings, "extra_filler_paired": paired_lengths}
    if a.b200_control:
        matched = unpack(a.b200_control)
        assert matched["fixture_sha256"] == source["fixture_sha256"] and matched["device"] == "NVIDIA B200"
        key = f"{matched['task']}/{matched['length']}"
        original = observations[key]
        assert set(original) == {row["id"] for row in matched["rows"]}
        equivalent = sum({k: v for k, v in row.items() if k != "key_seconds"} ==
                         {k: v for k, v in original[row["id"]].items() if k != "key_seconds"} for row in matched["rows"])
        result["b200_control"] = {"task_and_length": key, "matching_all_rank_records": equivalent,
                                 "total": len(original), "elapsed_seconds": matched["elapsed_seconds"]}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
