#!/usr/bin/env python3
"""Compare where stronger intermediate or final readers affect MuSiQue chaining."""

import argparse
import json
from pathlib import Path

from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from kam_flan_reader_assets import LARGE_REVISION

BASELINE = "top1"
METRICS = ("pool_support", "last_subanswer", "joint", "first_binding", "penultimate_binding",
           "penultimate_gold_context", "last_gold_context")


def switches(hops):
    arms = {BASELINE: (), "large_first": (0,), "large_final": (hops - 1,)}
    if hops > 2:
        arms |= {"large_penultimate": (hops - 2,), "large_all_intermediate": tuple(range(hops - 1))}
    return arms


def score(cases, outcomes):
    groups = shared_evidence_clusters(cases)
    flags = {}
    for arm in switches(cases[0]["hops"]):
        calls = []
        for q in cases:
            r = outcomes[q["id"]][arm]
            support = set(q["gold"]) <= set(r["pool"])
            final = bool(r["answer_em"][-1])
            calls.append((support, final, support and final, bool(r["answer_em"][0]), bool(r["answer_em"][-2]),
                          r["gold_in_reader"][-2], r["gold_in_reader"][-1]))
        flags[arm] = {key: list(v) for key, v in zip(METRICS, zip(*calls, strict=True), strict=True)}
    counts = {name: {k: sum(seq) for k, seq in r.items()} for name, r in flags.items()}
    pairs = {name: {k: compare(v, flags[BASELINE][k], groups) for k, v in r.items()}
             for name, r in flags.items() if name != BASELINE}
    return {"questions": len(cases), "clusters": len(groups), "counts": counts, "paired": pairs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--baseline", type=Path, nargs=4, required=True)
    p.add_argument("--trials", type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = unpack(a.manifest)
    cases = source["rows"]
    original, trials = [sorted(map(unpack, p), key=lambda r: r["shard"]) for p in (a.baseline, a.trials)]
    outcomes = {}
    for shard, (old, new) in enumerate(zip(original, trials, strict=True)):
        assert (old["shard"], new["shard"], old["shards"], new["shards"]) == (shard, shard, 4, 4)
        assert old["device"] == new["device"] == "NVIDIA L4"
        assert old["models"] | {"google/flan-t5-large": LARGE_REVISION} == new["models"]
        assert old["fixture_sha256"] == new["fixture_sha256"] == source["fixture_sha256"]
        assert old["corpus_sha256"] == new["corpus_sha256"] == source["corpus_sha256"]
        for key in ("templates_sha256", "scoring_labels_sha256", "baseline_sha256"):
            assert old[key] == new[key]
        for q, base, trial in zip(cases[shard::4], old["rows"], new["rows"], strict=True):
            id_, hops = q["id"], q["hops"]
            assert id_ == base["id"] == trial["id"] and id_ not in outcomes
            arms = trial["arms"]
            reference = base["arms"][BASELINE]
            assert arms[BASELINE] == reference and set(arms) == set(switches(hops))
            for name, steps in switches(hops).items():
                if name == BASELINE:
                    continue
                replacement = arms[name]
                assert len(replacement["pool"]) == len(set(replacement["pool"])) == 128
                assert replacement["pool"][:64] == reference["pool"][:64]
                first = min(steps)
                for key in ("target_ranks", "reader_context_ids", "gold_in_reader"):
                    assert replacement[key][:first+1] == reference[key][:first+1]
                assert replacement["answer_em"][:first] == reference["answer_em"][:first]
                if first == hops - 1:
                    assert replacement["pool"] == reference["pool"]
            outcomes[id_] = arms
    assert len(outcomes) == len(cases)
    result = {"fixture_sha256": source["fixture_sha256"], "baseline_exact_replays": len(cases),
              "scopes": {str(h): score([q for q in cases if q["hops"] == h], outcomes) for h in COHORTS}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
