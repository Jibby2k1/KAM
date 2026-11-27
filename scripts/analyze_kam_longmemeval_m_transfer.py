#!/usr/bin/env python3
"""Measure zero-shot Wikipedia decomposition on LongMemEval-M evidence."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

MODES = ("dense", "unresolved", "bound")


def outcomes(q, route):
    ids = q["source_session_ids"]
    marks = q["author_turn_ids"]
    picked = route["sessions"]
    gold = set(q["gold_ids"])
    assert len(picked) == len({ids[i] for i in picked}) == 5
    assert len(route["turns"]) <= 20 and all(i in picked for i, _ in route["turns"])
    retrieved = {ids[i] for i in picked}
    read = {ids[i] for i, place in route["turns"] if place in marks[i]}
    return {"all_sessions": gold <= retrieved, "gold_sessions": len(gold & retrieved),
            "any_marked_turn": bool(read), "marked_turn_per_gold_session": gold <= read}


def panel(cases, records):
    groups = shared_evidence_clusters([{"gold": q["gold_ids"]} for q in cases])
    result = {}
    for i in range(3):
        stats = {a: Counter() for a in MODES}
        flags = {a: {"all_sessions": [], "marked_turn_per_gold_session": []} for a in MODES}
        eligible = 0
        for q in cases:
            src = records[q["id"]]
            predicted = src["decomposers"][i]
            assert not predicted["valid"] or predicted["predicted_steps"] in (2, 3, 4)
            eligible += predicted["valid"]
            if not predicted["valid"]:
                assert predicted["unresolved"] == predicted["bound"] == src["dense"]
            selected = {"dense": src["dense"], **{a: predicted[a] for a in MODES[1:]}}
            for name, policy in selected.items():
                m = outcomes(q, policy)
                stats[name].update(m)
                for k in flags[name]:
                    flags[name][k].append(m[k])
        contrasts = {f"{a}_vs_{b}": {k: compare(flags[a][k], flags[b][k], groups) for k in flags[a]}
                     for a, b in (("unresolved", "dense"), ("bound", "dense"), ("bound", "unresolved"))}
        result[str(i)] = {"eligible_plans": eligible, "metrics": stats, "paired": contrasts}
    return {"questions": len(cases), "scoring": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    for mode in ("reference", "transfer"):
        p.add_argument("--" + mode, nargs=4, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    f = unpack(a.manifest)
    by_hash = {sha256(file.read_bytes()).hexdigest(): unpack(file) for file in a.reference}
    rows = sorted(map(unpack, a.transfer), key=lambda r: r["shard"])
    assert [r["shard"] for r in rows] == list(range(4))
    common = ("fixture_sha256", "plans_sha256", "seeds", "models")
    out = {}
    for chunk in rows:
        r = by_hash[chunk["reference_sha256"]]
        assert all(rows[0][k] == chunk[k] for k in common)
        assert r["shard"] == chunk["shard"] and r["device"] == chunk["device"] == "NVIDIA L4"
        assert r["fixture_sha256"] == chunk["fixture_sha256"] == f["fixture_sha256"]
        for baseline, result, q in zip(r["rows"], chunk["rows"], f["rows"][chunk["shard"]::4], strict=True):
            assert q["id"] == baseline["id"] == result["id"]
            assert result["dense"]["sessions"] == baseline["full"]["dense"][:5]
            assert result["dense"]["turns"] == baseline["turns_from_top5"]["dense"]
            out[result["id"]] = result
    cases = [q for q in f["rows"] if not q["id"].endswith("_abs")]
    assert len(cases) == 470 and len(out) == 500
    scopes = {"all": cases, **{k: [q for q in cases if q["kind"] == k] for k in sorted({q["kind"] for q in cases})}}
    record = {"fixture_sha256": f["fixture_sha256"], "seeds": rows[0]["seeds"], "plans_sha256": rows[0]["plans_sha256"],
              "scopes": {key: panel(s, out) for key, s in scopes.items()}}
    a.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
