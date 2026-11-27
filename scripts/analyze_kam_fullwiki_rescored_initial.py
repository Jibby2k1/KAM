#!/usr/bin/env python3
"""Measure live first-hop rescoring at two fixed conditioned search policies."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles

import numpy as np

from analyze_kam_fullwiki_coarse_live import PHASES
from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

SECOND = ("full_rescore", "fused_256")


def compare(rows, fixture, second):
    raw, candidate = (f"{method}_initial_{second}" for method in ("exact", "rescore"))
    matches, counts = Counter(), {m: Counter() for m in (raw, candidate)}
    times = {m: [] for m in (raw, candidate)}
    phases = {m: {key: [] for key in PHASES} for m in (raw, candidate)}
    flags = {m: {"complete_original": [], "support": [], "joint": [], "answer": []} for m in (raw, candidate)}
    for case in rows:
        original, near = (case["records"][m] for m in (raw, candidate))
        expected = set(fixture[case["id"]]["gold_ids"])
        exact, recv = (q["arms"]["conditioned_union"] for q in (original, near))
        matches.update({"initial_order": original["initial_ids"] == near["initial_ids"],
                        "initial_set": set(original["initial_ids"]) == set(near["initial_ids"]),
                        "anchor": original["first"] == near["first"],
                        "conditioned_order": original["corpus_hits"] == near["corpus_hits"],
                        "pair_order": exact["pair"] == recv["pair"], "outcome": exact == recv})
        for method in flags:
            item = case["records"][method]
            outcome = item["arms"]["conditioned_union"]
            support = set(outcome["pair"]) == expected
            answers = bool(outcome["exact"])
            status = {"complete_original": expected <= set(item["initial_ids"]),
                      "support": support, "answer": answers, "joint": support and answers}
            for k, value in status.items():
                flags[method][k].append(value)
            counts[method].update(status)
            times[method].append(case["request_ms"][method])
            for k in PHASES:
                phases[method][k].append(case["phases_ms"][method][k])
    ranges = {m: {"request_p50_ms": median(v), "request_p95_ms": quantiles(v, n=100, method="inclusive")[94],
                  "median_phase_ms": {key: median(phases[m][key]) for key in PHASES}} for m, v in times.items()}
    differences = np.asarray(times[candidate]) - times[raw]
    sampling = np.random.default_rng(73331).integers(0, len(rows), size=(4000, len(rows)))
    return {"requests": len(rows), "identical_vs_first_exact": matches, "counts": counts, "latency": ranges,
            "paired_quality_change": {k: paired(x, flags[raw][k]) for k, x in flags[candidate].items()},
            "paired_median_ms": float(np.median(differences)),
            "paired_95_ms": np.quantile(np.median(differences[sampling], axis=1), [0.025, 0.975]).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, nargs=4, required=True)
    p.add_argument("--online", type=Path, nargs=4, required=True)
    a = p.parse_args()
    raw = gzip.decompress(a.fixture.read_bytes())
    fixture = {q["id"]: q for q in json.loads(raw)["rows"]["test"]}
    hash_ = sha256(raw).hexdigest()
    old, new = (sorted(map(unpack, q), key=lambda c: c["shard"]) for q in (a.reference, a.online))
    assert [c["shard"] for c in old] == [c["shard"] for c in new] == list(range(4))
    arms = {f"{first}_initial_{second}" for first in ("exact", "rescore") for second in SECOND}
    for past, now in zip(old, new, strict=True):
        assert [q["id"] for q in past["rows"]] == [q["id"] for q in now["rows"]]
        assert all(past[k] == now[k] for k in ("fixture_sha256", "index_sha256", "corpus_profile_partition", "models"))
        assert now["fixture_sha256"] == hash_ and now["gpu"] == past["gpu"] == "NVIDIA L4"
        assert set(now["search_modes"]) == arms and len(now["rows"]) == 48
    rows = [q for c in new for q in c["rows"]]
    assert len(rows) == len({q["id"] for q in rows}) == 192
    result = {"fixture_sha256": hash_, "second_hop": {arm: compare(rows, fixture, arm) for arm in SECOND}}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
