#!/usr/bin/env python3
"""Compare added decomposition candidates with equally deep dense retrieval."""

import argparse
import json
from pathlib import Path

from analyze_kam_longmemeval_m_transfer import read_panel
from analyze_kam_native_musique import compare, shared_evidence_clusters

DEPTHS = (6, 8, 10)


def additions(ids, primary, alternate, depth):
    found = dict.fromkeys(ids[i] for i in primary[:5])
    for item in alternate + primary[5:]:
        if len(found) == depth:
            break
        found[ids[item]] = None
    assert len(found) == depth
    return set(found)


def score(cases, baseline, transfer):
    groups = shared_evidence_clusters([{"gold": q["gold_ids"]} for q in cases])
    result = {}
    for depth in DEPTHS:
        original = []
        expanded = [{name: [] for name in ("unresolved", "bound")} for _ in range(3)]
        for q in cases:
            key = q["id"]
            ids = q["source_session_ids"]
            rank = baseline[key]["full"]["dense"]
            dense = {ids[i] for i in rank[:depth]}
            assert len(dense) == depth
            gold = set(q["gold_ids"])
            original.append(gold <= dense)
            for flags, proposal in zip(expanded, transfer[key]["decomposers"], strict=True):
                for arm in flags:
                    candidates = additions(ids, rank, proposal[arm]["sessions"], depth)
                    assert all(ids[i] in candidates for i in rank[:5])
                    flags[arm].append(gold <= candidates)
        result[str(depth)] = {"dense": sum(original), "decomposition": [
            {arm: {"all_sessions": sum(flags), "vs_dense": compare(flags, original, groups)}
             for arm, flags in seed.items()} for seed in expanded]}
    return {"questions": len(cases), "depths": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    for mode in ("reference", "transfer"):
        p.add_argument("--" + mode, nargs=4, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fixture, cases, baseline, transfer, meta = read_panel(a.manifest, a.reference, a.transfer)
    scopes = {"all": cases, "temporal": [q for q in cases if q["kind"] == "temporal-reasoning"],
              "multi_session": [q for q in cases if len(q["gold_ids"]) > 1]}
    payload = {"fixture_sha256": fixture["fixture_sha256"], "seeds": meta["seeds"],
               "scopes": {name: score(group, baseline, transfer) for name, group in scopes.items()}}
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
