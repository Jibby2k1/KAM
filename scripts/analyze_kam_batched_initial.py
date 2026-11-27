#!/usr/bin/env python3
"""Compare shared and separate initial corpus scans in matched request batches."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from analyze_kam_native_musique import compare
from analyze_kam_shared_serving import unpack

ARMS = ("separate", "shared")
PROFILES = {"reader_precision": ("fp32", "bf16"), "prompt_cache": ("baseline", "cached_prompt"),
            "feature_reuse": ("baseline", "shared_features"), "second_batch": ("baseline", "shared_second")}
STAGES = ("request_ms", "initial_search_ms", "reader_ms")
DETAILED = ("initial_encode_ms", "initial_articles_ms", "anchor_ms", "conditioned_search_ms",
            "rerank_prepare_ms", "rerank_score_ms", "prompt_build_ms", "remaining_rerank_ms")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in ("measurement", "reference"):
        p.add_argument("--" + name, type=Path, nargs=4, required=True)
    p.add_argument("--device", choices=("NVIDIA L4", "NVIDIA B200"), default="NVIDIA L4")
    comparisons = p.add_mutually_exclusive_group()
    for name in PROFILES:
        comparisons.add_argument("--compare-" + name.replace("_", "-"), dest="profile", action="store_const", const=name)
    a = p.parse_args()
    arms = PROFILES.get(a.profile, ARMS)
    reference_arm = "shared" if a.profile else "fp16/8"
    detailed = a.profile in ("prompt_cache", "feature_reuse", "second_batch")
    reference_device = a.device if detailed else "NVIDIA L4"
    reference_name = "baseline_vs_archive" if detailed else ("fp32_vs_archive" if a.profile else "separate_vs_archive")
    stages = STAGES + DETAILED if detailed else STAGES
    raw = gzip.decompress(a.fixture.read_bytes())
    fx = sha256(raw).hexdigest()
    truth = {x["id"]: set(x["gold_ids"]) for x in json.loads(raw)["rows"]["test"]}
    experiments, replay = [sorted(map(unpack, args), key=lambda c: c["shard"]) for args in (a.measurement, a.reference)]
    times = {m: {phase: [] for phase in stages} for m in arms}
    prompt_work = {m: Counter() for m in arms}
    flags = {m: {k: [] for k in ("support", "answer", "joint")} for m in arms}
    parity = Counter()
    distinct = set()
    for i, (done, baseline) in enumerate(zip(experiments, replay, strict=True)):
        assert done["fixture_sha256"] == baseline["fixture_sha256"] == fx
        assert done["shard"] == baseline["shard"] == i and done["shards"] == baseline["shards"] == 4
        assert all(done[k] == baseline[k] for k in ("models", "index_sha256", "fitted_sha256", "corpus_profile_partition"))
        assert done["gpu"] == a.device and baseline["gpu"] == reference_device
        assert len(done["rows"]) == len(baseline["rows"]) == 25 and set(done["batch_policies"]) == set(arms)
        for trial, anchor in zip(done["rows"], baseline["rows"], strict=True):
            assert trial["ids"] == anchor["ids"] and len(trial["ids"]) == 8
            assert set(trial["order"]) == set(trial["batches"]) == set(arms)
            assert distinct.isdisjoint(trial["ids"])
            distinct.update(trial["ids"])
            separate, shared = [trial["batches"][m] for m in arms]
            for name, stage in zip(arms, (separate, shared), strict=True):
                timing, = stage["timing"]
                assert timing["size"] == 8
                for phase in stages:
                    times[name][phase].append(timing[phase])
                if a.profile == "prompt_cache":
                    assert timing["passage_tokenized"] <= timing["passage_requests"]
                    prompt_work[name].update({k: timing[k] for k in ("passage_requests", "passage_tokenized")})
            if detailed:
                control, candidate = (arm["timing"][0] for arm in (separate, shared))
                parity["prompt_batches"] += control["prompts_sha256"] == candidate["prompts_sha256"]
                if a.profile == "feature_reuse":
                    parity["full_rerank_feature_batches"] += control["rerank_features_sha256"] == candidate["rerank_features_sha256"]
            before = anchor["batches"][reference_arm]["rows"]
            for ref, one, many, id_ in zip(before, separate["rows"], shared["rows"], trial["ids"], strict=True):
                assert ref["id"] == one["id"] == many["id"] == id_
                selected = [r["arms"]["conditioned_union"]["pair"] for r in (one, many)]
                if a.profile == "reader_precision":
                    parity.update({"outside_reader": all(one[k] == many[k] for k in one if k != "arms"),
                                   "both_evidence_pairs": all(one["arms"][k]["pair"] == many["arms"][k]["pair"] for k in one["arms"]),
                                   "both_answer_flags": all(one["arms"][k]["exact"] == many["arms"][k]["exact"] for k in one["arms"]),
                                   "conditioned_answer_flag": one["arms"]["conditioned_union"]["exact"] == many["arms"]["conditioned_union"]["exact"]})
                parity.update({reference_name: one == ref, "arms_vs_archive": one["arms"] == ref["arms"],
                               "initial_set": set(one["initial_ids"]) == set(many["initial_ids"]),
                               "initial_order": one["initial_ids"] == many["initial_ids"],
                               "second_hop": one["corpus_hits"] == many["corpus_hits"],
                               "conditioned_pairs": selected[0] == selected[1], "all_records": one == many})
                for name, out, pair in zip(arms, (one, many), selected, strict=True):
                    support = set(pair) == truth[id_]
                    answer = out["arms"]["conditioned_union"]["exact"]
                    for key, hit in zip(flags[name], (support, answer, support and answer), strict=True):
                        flags[name][key].append(hit)
    n = len(distinct)
    assert n == 800
    sample = np.random.default_rng(720528).integers(0, 100, size=(4000, 100))
    summaries = {}
    for phase in stages:
        old, new = [np.asarray(times[arm][phase]) for arm in arms]
        assert len(old) == len(new) == 100
        delta = new - old
        summaries[phase] = {"p50_ms": dict(zip(arms, np.median([old, new], axis=1).tolist(), strict=True)),
                            "p95_ms": dict(zip(arms, np.quantile([old, new], .95, axis=1).tolist(), strict=True)),
                            "paired_median_delta_ms": float(np.median(delta)),
                            "paired_median_delta_95_ms": np.quantile(np.median(delta[sample], axis=1), [.025, .975]).tolist()}
    result = {"fixture_sha256": fx, "device": a.device, "requests": n, "parity": parity, "latency": summaries,
              "requests_per_second": {m: 1000*n/sum(times[m]["request_ms"]) for m in arms},
              "counts": {m: {k: sum(row) for k, row in f.items()} for m, f in flags.items()},
              "quality": {k: compare(flags[arms[1]][k], v, [[i] for i in range(n)]) for k, v in flags[arms[0]].items()}}
    if a.profile == "prompt_cache":
        result["passage_work"] = prompt_work
    if a.profile in ("prompt_cache", "feature_reuse"):
        assert parity["prompt_batches"] == 100 and parity["all_records"] == n
    if a.profile == "feature_reuse":
        assert parity["full_rerank_feature_batches"] == 100
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
