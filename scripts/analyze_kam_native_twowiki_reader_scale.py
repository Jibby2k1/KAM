#!/usr/bin/env python3
"""Measure paired exact match with reader capacity and prompt inputs audited."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_native_twowiki import GROUPS, load_shards, verify_fixture
from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

ARMS = ("frozen", "refit", "gold_native", "gold_annotations")
MODELS = ("small", "base", "base_bf16")


def assess(cases, runs):
    totals = {}
    for arm in ARMS:
        by_model = {k: [rows[q["id"]]["arms"][arm] for q in cases] for k, rows in runs.items()}
        earlier = by_model["small"]
        assert all(a["support"] == b["support"] and a["archived_small_em"] == b["archived_small_em"]
                   for seq in by_model.values() for a, b in zip(earlier, seq, strict=True))
        support = [x["support"] for x in earlier]
        em = {name: [bool(a["em"]) for a in entries] for name, entries in by_model.items()}
        joint = {k: [a and b for a, b in zip(v, support, strict=True)] for k, v in em.items()}
        comparisons = [("base", "small"), ("base_bf16", "small"), ("base_bf16", "base")]
        totals[arm] = {"support": sum(support), "exact": {k: sum(v) for k, v in em.items()},
                       "joint": {k: sum(v) for k, v in joint.items()},
                       "archived_em_mismatches": sum(bool(a["archived_small_em"]) != b for a, b in zip(earlier, em["small"])),
                       "paired_em": {f"{a}_vs_{b}": paired(em[a], em[b]) for a, b in comparisons},
                       "paired_joint": {f"{a}_vs_{b}": paired(joint[a], joint[b]) for a, b in comparisons}}
    return {"questions": len(cases), "arms": totals}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    for arm in (*MODELS, "fixed", "refit"):
        p.add_argument("--" + arm, required=True, nargs=4, type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    f = unpack(a.manifest)
    verify_fixture(f)
    records = {k: load_shards(getattr(a, k), f) for k in MODELS}
    chunks = {k: sorted(map(unpack, getattr(a, k)), key=lambda s: s["shard"]) for k in MODELS}
    source_file_hashes = {k: {sha256(p.read_bytes()).hexdigest() for p in v}
                          for k, v in (("pairs_sha256", a.refit), ("fixed_sha256", a.fixed))}
    for model, seq in chunks.items():
        for x, y in zip(chunks["small"], seq, strict=True):
            assert y["model"] == model
            assert x["source_sha256"] == y["source_sha256"] == f["source_sha256"]
            assert all(x[k] == y[k] for k in ("prompt_sha256", "token_sha256", "prompts"))
            assert all(x[k] == y[k] and x[k] in values for k, values in source_file_hashes.items())
    cases = f["rows"]
    groups = {"all": cases, **{g: [q for q in cases if q["cohort"] == g] for g in GROUPS}}
    states = Counter()
    for arm, label in (("frozen", "frozen_only"), ("refit", "refit_only")):
        other = "refit" if arm == "frozen" else "frozen"
        groups[label] = [q for q in cases if records["small"][q["id"]]["arms"][arm]["support"]
                         and not records["small"][q["id"]]["arms"][other]["support"]]
        states[label] = len(groups[label])
    out = {"fixture_sha256": f["fixture_sha256"], "discordant_support": states,
           "runtime": {k: {"revision": seq[0]["revision"], "parameters": seq[0]["parameters"],
                                        "prompts": [x["prompts"] for x in seq],
                                        "decode_seconds": [x["decode_seconds"] for x in seq]} for k, seq in chunks.items()},
           "scopes": {name: assess(g, records) for name, g in groups.items()}}
    a.output.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
