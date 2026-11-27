#!/usr/bin/env python3
"""Verify paired support and joint exact match on the fixed 2Wiki shortlist."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_native_twowiki import GROUPS, load_shards, verify_fixture
from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

ARMS = ("frozen", "refit")


def assess(cases, outcomes, rankings, reader_controls):
    stage = {arm: Counter() for arm in ARMS}
    hits = {arm: {"support": [], "joint": [], "answer_em": []} for arm in ARMS}
    changed_pairs = 0
    recovered_reader_controls = {name: Counter() for name in ("frozen_only", "refit_only", "both")}
    for item in cases:
        id_ = item["id"]
        row = outcomes[id_]
        gold = {u["corpus_id"] for u in item["units"]}
        pool = set(rankings[id_]["conditioned"])
        complete, shortlisted = gold <= pool, row["shortlisted"]
        changed_pairs += row["frozen"]["pair"] != row["refit"]["pair"]
        for arm in ARMS:
            choice = row[arm]
            pair = set(choice["pair"])
            assert len(pair) == 2 and pair <= pool
            support, answer = pair == gold, bool(choice["em"])
            assert support <= shortlisted <= complete
            joint = support and answer
            stage[arm].update({"pool": complete, "shortlist": shortlisted,
                               "support": support, "joint": joint, "answer_em": answer})
            hits[arm]["support"].append(support)
            hits[arm]["joint"].append(joint)
            hits[arm]["answer_em"].append(answer)
        frozen_support, refit_support = (hits[a]["support"][-1] for a in ARMS)
        if frozen_support or refit_support:
            group = "both" if frozen_support and refit_support else ("frozen_only" if frozen_support else "refit_only")
            stats = recovered_reader_controls[group]
            stats.update({"questions": 1, "frozen_joint": hits["frozen"]["joint"][-1],
                          "refit_joint": hits["refit"]["joint"][-1]})
            for method, control in reader_controls[id_]["arms"].items():
                stats[method] += control["em"]
    return {"questions": len(cases), "changed_pairs": changed_pairs, "counts": stage,
            "recovered_reader_controls": recovered_reader_controls,
            "paired_refit_vs_frozen": {key: paired(hits["refit"][key], hits["frozen"][key])
                                      for key in hits["frozen"]},
            "oracle_policy_choice": {key: sum(a or b for a, b in zip(hits["refit"][key], hits["frozen"][key], strict=True))
                                     for key in ("support", "joint")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--search", nargs=4, required=True, type=Path)
    p.add_argument("--reader", nargs=4, required=True, type=Path)
    p.add_argument("--fixed-reader", nargs=4, required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    fixture = unpack(a.manifest)
    verify_fixture(fixture)
    ledger = sorted(sha256(file.read_bytes()).hexdigest() for file in a.search)
    policy_hash = sha256(a.policy.read_bytes()).hexdigest()
    chunks = list(map(unpack, a.reader))
    assert len({s["picks_sha256"] for s in chunks}) == 1
    assert all(c["search_hashes"] == ledger and c["policy_sha256"] == policy_hash and c["device"] == "NVIDIA L4"
               and c["replayed_frozen_scores"] == 3 * len(c["rows"]) for c in chunks)
    rankings, outcomes = load_shards(a.search, fixture), load_shards(a.reader, fixture)
    controls = load_shards(a.fixed_reader, fixture)
    assert all(unpack(p)["models"] == chunks[0]["models"] for p in a.fixed_reader)
    cases = fixture["rows"]
    groups = {"all": cases, **{g: [q for q in cases if q["cohort"] == g] for g in GROUPS}}
    for status in (True, False):
        name = "dense64_complete" if status else "dense64_incomplete"
        groups[name] = [q for q in cases
                        if ({u["corpus_id"] for u in q["units"]} <= set(rankings[q["id"]]["dense"][:64])) == status]
    result = {"policy_sha256": policy_hash, "picks_sha256": chunks[0]["picks_sha256"],
              "fixture_sha256": fixture["fixture_sha256"],
              "reader_replay": {"frozen_scores": sum(s["replayed_frozen_scores"] for s in chunks),
                                "new_prompts": sum(s["new_reader_prompts"] for s in chunks)},
              "scopes": {scope: assess(rows, outcomes, rankings, controls) for scope, rows in groups.items()}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
