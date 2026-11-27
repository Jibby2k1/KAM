#!/usr/bin/env python3
"""Verify the locked pair-and-reader replay on frozen native 2Wiki lists."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_native_twowiki import GROUPS, load_shards, verify_fixture
from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

ARMS = ("dense64", "dense128", "conditioned128")
FITTED = "0abfd208cecf86133ffeaf581996b7a6bb6686ed28d9f0ac73eaef8966d3636a"
METRICS = ("support", "joint", "answer_em")


def cases(candidates, search, rerank):
    totals = {a: Counter() for a in ARMS}
    scores = {a: {metric: [] for metric in METRICS} for a in ARMS}
    for question in candidates:
        i = question["id"]
        row, runs = rerank[i], search[i]
        gold = {u["corpus_id"] for u in question["units"]}
        assert set(row["arms"]) == set(ARMS)
        for name, pool in zip(ARMS, (runs["dense"][:64], runs["dense"], runs["conditioned"]), strict=True):
            a = row["arms"][name]
            pair = set(a["pair"])
            assert len(pair) == 2 and pair <= set(pool)
            cover = gold <= set(pool)
            selected = pair == gold
            chosen = bool(a["em"])
            assert selected <= a["shortlisted"] <= cover
            totals[name].update({"pool": cover, "shortlisted": a["shortlisted"],
                                 "support": selected, "answer_em": chosen, "joint": selected and chosen})
            for key, value in zip(METRICS, (selected, selected and chosen, chosen), strict=True):
                scores[name][key].append(value)
    diffs = {f"{b}_vs_{a}": {m: paired(scores[b][m], scores[a][m]) for m in METRICS}
             for a, b in (("dense64", "dense128"), ("dense64", "conditioned128"), ("dense128", "conditioned128"))}
    oracles = {name: {m: sum(any(xs) for xs in zip(*(scores[a][m] for a in arms), strict=True))
                     for m in ("support", "joint")}
               for name, arms in (("dense64_or_conditioned", ("dense64", "conditioned128")),
                                  ("all_three", ARMS))}
    stages = {}
    for k, totals_for_arm in totals.items():
        n = len(candidates)
        pool, shortlist, support, jointly = (totals_for_arm[s] for s in ("pool", "shortlisted", "support", "joint"))
        stages[k] = {"outside_pool": n - pool, "outside_shortlist": pool - shortlist, "pair_rank": shortlist - support,
                     "reader": support - jointly, "joint": jointly}
    return {"questions": len(candidates), "counts": totals, "stage_counts": stages, "paired": diffs,
            "oracle_policy_choice": oracles}


def summarize(fixture, search, rerank):
    verify_fixture(fixture)
    all_rows = fixture["rows"]
    groups = {"all": all_rows}
    for cohort in GROUPS:
        grouped = groups[cohort] = [q for q in all_rows if q["cohort"] == cohort]
        complete, incomplete = [], []
        for q in grouped:
            in_original = {u["corpus_id"] for u in q["units"]} <= set(search[q["id"]]["dense"][:64])
            (complete if in_original else incomplete).append(q)
        groups[f"{cohort}/dense64_complete"] = complete
        groups[f"{cohort}/dense64_incomplete"] = incomplete
    return {"fixture_sha256": fixture["fixture_sha256"], "fitted_sha256": FITTED,
            "groups": {block: cases(q, search, rerank) for block, q in groups.items() if q}}


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--search", nargs=4, required=True, type=Path)
    a.add_argument("--rerank", nargs=4, required=True, type=Path)
    a.add_argument("--manifest", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)
    cfg = a.parse_args()
    fixture = unpack(cfg.manifest)
    search_hashes = sorted(sha256(p.read_bytes()).hexdigest() for p in cfg.search)
    for p in cfg.rerank:
        s = unpack(p)
        assert s["search_hashes"] == search_hashes and s["fitted_sha256"] == FITTED
        assert s["source_sha256"] == fixture["source_sha256"]
    result = summarize(fixture, load_shards(cfg.search, fixture), load_shards(cfg.rerank, fixture))
    cfg.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
