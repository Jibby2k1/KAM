#!/usr/bin/env python3
"""Audit matched LongMemEval context, clipping and generation budgets."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from run_kam_longmemeval_m_reader import CONTEXT_VARIANTS

PAIRS = (("dense_512_window", "dense_512_prefix"), ("dense_1024_prefix", "dense_512_prefix"),
         ("dense_1024_window", "dense_1024_prefix"), ("dense_1536_prefix", "dense_1024_prefix"),
         ("dense_1024_long_answer", "dense_1024_prefix"), ("oracle_1024_prefix", "oracle_512_prefix"),
         ("oracle_1024_long_answer", "oracle_1024_prefix"))


def summarize(cases, records):
    components = shared_evidence_clusters([{"gold": q["gold_ids"] or [q["id"]]} for q in cases])
    flags, trials = {}, {}
    for arm in CONTEXT_VARIANTS:
        rows = [records[q["id"]][arm] for q in cases]
        flags[arm] = {"answer_em": [r["em"] for r in rows], "joint_em": [r["em"] and r["marked_support"] for r in rows],
                      "abstention": [r["abstained"] for r in rows]}
        trials[arm] = {k: sum(value) for k, value in flags[arm].items()} | {
            "mean_f1": mean(r["f1"] for r in rows), "median_prompt_tokens": median(r["prompt_tokens"] for r in rows),
            "median_retained_text": median(r["retained_turn_tokens"] for r in rows),
            "clipped_questions": sum(r["clipped_turns"] > 0 for r in rows),
            "questions_with_shifted_turns": sum(r["shifted_turns"] > 0 for r in rows)}
    return {"questions": len(cases), "results": trials,
            "paired": {f"{a}_vs_{b}": {k: compare(flags[a][k], flags[b][k], components) for k in flags[a]} for a, b in PAIRS}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("reader", "original"):
        p.add_argument("--" + name, type=Path, nargs=4, required=True)
    a = p.parse_args()
    manifest = unpack(a.manifest)
    cases = manifest["rows"]
    new, prior = [sorted(map(unpack, inputs), key=lambda c: c["shard"]) for inputs in (a.reader, a.original)]
    assert [c["shard"] for c in new] == list(range(4))
    equal, records = Counter(), {}
    keys = ("fixture_sha256", "reference_sha256", "labels_sha256", "reader_revision", "reader_weights_sha256", "device")
    for trial, base in zip(new, prior, strict=True):
        assert trial["shard"] == base["shard"] and trial["shards"] == base["shards"] == 4
        assert trial["fixture_sha256"] == manifest["fixture_sha256"] and trial["device"] == "NVIDIA L4"
        assert all(trial[k] == base[k] for k in keys) and set(trial["arms"]) == set(CONTEXT_VARIANTS)
        for q, record, legacy in zip(cases[trial["shard"]::4], trial["rows"], base["rows"], strict=True):
            assert record["id"] == legacy["id"] == q["id"]
            arm = record["arms"]
            for name, (fixed, cap, shifted, _) in CONTEXT_VARIANTS.items():
                orig, got = legacy["arms"][fixed], arm[name]
                assert all(orig[k] == got[k] for k in ("turns", "original_turn_tokens", "marked_support"))
                assert got["prompt_tokens"] <= cap and got["shifted_turns"] <= got["clipped_turns"]
                assert shifted or not got["shifted_turns"]
                if cap == 512 and not shifted:
                    equal[name] += {k: v for k, v in got.items() if k != "shifted_turns"} == orig
            for name in ("dense_512", "dense_1024"):
                assert arm[name + "_window"]["retained_turn_tokens"] == arm[name + "_prefix"]["retained_turn_tokens"]
            for name in ("dense_1024", "oracle_1024"):
                prefix, extended = arm[name + "_prefix"], arm[name + "_long_answer"]
                assert all(prefix[k] == extended[k] for k in ("retained_turn_tokens", "prompt_tokens", "turns"))
            records[q["id"]] = arm
    eligible = [q for q in cases if not q["id"].endswith("_abs")]
    groups = {"answerable": eligible, "abstention": [q for q in cases if q["id"].endswith("_abs")]}
    for category in sorted({q["kind"] for q in eligible}):
        groups[category] = [q for q in eligible if q["kind"] == category]
    assert len(records) == len(cases) == 500
    result = {"fixture_sha256": manifest["fixture_sha256"], "parity_with_original_512": equal,
              "groups": {name: summarize(q, records) for name, q in groups.items()}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
