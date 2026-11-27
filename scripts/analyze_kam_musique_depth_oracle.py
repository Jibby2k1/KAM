#!/usr/bin/env python3
"""Separate missing decomposition steps from subquestion quality on MuSiQue."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(paths, manifest, weighted, oracle):
    chunks = sorted(map(unpack, paths), key=lambda p: p["shard"])
    assert [c["shard"] for c in chunks] == list(range(4))
    cases = manifest["rows"]
    out = {}
    for c in chunks:
        assert bool(c.get("oracle_depth")) == oracle
        assert c["device"] == "NVIDIA L4" and c["weights_sha256"] == weighted and c["shards"] == 4
        assert c["fixture_sha256"] == manifest["fixture_sha256"] and c["corpus_sha256"] == manifest["corpus_sha256"]
        for q, p in zip(cases[c["shard"]::4], c["rows"], strict=True):
            assert p["id"] == q["id"] and len(p["pool"]) == len(set(p["pool"])) == 128
            if oracle and not p["fallback"]:
                assert p["predicted_steps"] == q["hops"]
            out[p["id"]] = p
    return out


def assess(cases, normal, forced):
    totals = {m: Counter() for m in ("unconstrained", "forced")}
    flags = {k: {"complete": [], "valid": [], "correct_steps": []} for k in totals}
    for q in cases:
        gold = set(q["gold"])
        for name, rank in (("unconstrained", normal), ("forced", forced)):
            row = rank[q["id"]]
            visible = gold.intersection(row["pool"])
            info = {"complete": visible == gold, "valid": not row["fallback"],
                    "correct_steps": row["predicted_steps"] == q["hops"]}
            totals[name].update(info | {"par": len(visible), "short": row["predicted_steps"] < q["hops"]})
            for key, value in info.items():
                flags[name][key].append(value)
    groups = shared_evidence_clusters(cases)
    return {"questions": len(cases), "counts": totals,
            "paired": {k: compare(flags["forced"][k], v, groups) for k, v in flags["unconstrained"].items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("unconstrained", "oracle"):
        p.add_argument("--" + name, type=Path, nargs=4, action="append", required=True)
    p.add_argument("--oracle-plan", type=Path, nargs=3, required=True)
    a = p.parse_args()
    fixture = unpack(a.manifest)
    cases = fixture["rows"]
    scopes = {"all": cases, **{str(i): [q for q in cases if q["hops"] == i] for i in COHORTS}}
    scores = {}
    assert len(a.unconstrained) == len(a.oracle) == 3
    for idx, (lhs, rhs, plan) in enumerate(zip(a.unconstrained, a.oracle, a.oracle_plan, strict=True)):
        depth = unpack(plan)
        assert depth["oracle_depth"] and depth["fixture_sha256"] == fixture["fixture_sha256"]
        assert all(unpack(p)["generation_sha256"] == digest(plan) for p in rhs)
        orig, constrained = (read(p, fixture, depth["weights_sha256"], oracle)
                             for p, oracle in ((lhs, False), (rhs, True)))
        assert [x["id"] for x in depth["rows"]] == [q["id"] for q in cases]
        assert all(x["valid"] != constrained[x["id"]]["fallback"] for x in depth["rows"])
        data = {k: assess(v, orig, constrained) for k, v in scopes.items()}
        data["generation_sha256"] = digest(plan)
        scores[str(idx)] = data
    result = {"fixture_sha256": fixture["fixture_sha256"], "seeds": scores}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
