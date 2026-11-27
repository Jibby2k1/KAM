#!/usr/bin/env python3
"""Audit measured live request time and article-pair fidelity by search arm."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles

import numpy as np

from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

PHASES = ("initial_search_ms", "anchor_ms", "conditioned_search_ms", "remaining_rerank_ms", "reader_ms")


def summarize(rows, questions, methods):
    values, flags, measures, identity = {}, {}, {}, {}
    complete_before = [set(questions[r["id"]]["gold_ids"]) <= set(r["records"]["exact"]["initial_ids"]) for r in rows]
    for name in methods:
        errors, phases = Counter(), {key: [] for key in PHASES}
        timings, labels = [], {m: [] for m in ("pool", "support", "answer", "joint")}
        for item in rows:
            base = item["records"]["exact"]
            choice = item["records"][name]
            assert choice["initial_ids"] == base["initial_ids"] and choice["first"] == base["first"]
            assert choice["pool_hash"] == base["pool_hash"] and choice["arms"]["broad"]["pair"] == base["arms"]["broad"]["pair"]
            pool = choice["initial_ids"]
            conditioned = choice["corpus_hits"]["conditioned_union"]
            r = choice["arms"]["conditioned_union"]
            p = base["arms"]["conditioned_union"]
            assert len(pool) == len(set(pool)) == 64 and len(conditioned) == len(set(conditioned)) == 64
            assert len(r["pair"]) == len(set(r["pair"])) == 2 and set(r["pair"]) <= set(pool) | set(conditioned)
            gold = set(questions[item["id"]]["gold_ids"])
            complete = gold <= set(pool).union(conditioned)
            hit = gold == set(r["pair"])
            assert hit <= r["gold_pair_in_shortlist"] <= complete
            labels["pool"].append(complete)
            labels["support"].append(hit)
            labels["answer"].append(bool(r["exact"]))
            labels["joint"].append(hit and bool(r["exact"]))
            errors.update({"rank_exact": conditioned == base["corpus_hits"]["conditioned_union"],
                           "pair_exact": r["pair"] == p["pair"], "outcome_exact": r == p})
            timings.append(item["request_ms"][name])
            for key in PHASES:
                phases[key].append(item["phases_ms"][name][key])
        flags[name] = labels
        values[name] = timings
        measures[name] = {"p50_ms": median(timings), "p95_ms": quantiles(timings, n=100, method="inclusive")[94],
                          "phase_p50_ms": {name: median(scored) for name, scored in phases.items()}}
        identity[name] = errors
    rng = np.random.default_rng(407813)
    indices = rng.integers(0, len(rows), size=(4000, len(rows)))
    for name in methods:
        difference = np.asarray(values[name]) - values["exact"]
        q = np.quantile(np.median(difference[indices], axis=1), [0.025, 0.975])
        measures[name]["paired_median_delta_to_exact_ms"] = float(np.median(difference))
        measures[name]["paired_median_bootstrap95_ms"] = q.tolist()
    return {"questions": len(rows), "initial_complete": sum(complete_before), "timing": measures,
            "fidelity_vs_exact": identity,
            "counts": {name: {k: sum(seq) for k, seq in labels.items()} for name, labels in flags.items()},
            "paired_quality_vs_exact": {n: {k: paired(v, flags["exact"][k]) for k, v in f.items()}
                                        for n, f in flags.items() if n != "exact"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--online", type=Path, nargs=4, required=True)
    a = p.parse_args()
    raw_fixture = gzip.decompress(a.fixture.read_bytes())
    fixture_hash = sha256(raw_fixture).hexdigest()
    questions = json.loads(raw_fixture)["rows"]["test"]
    chunks = sorted(map(unpack, a.online), key=lambda r: r["shard"])
    assert [c["shard"] for c in chunks] == list(range(4))
    frozen = chunks[0]
    keys = ("fixture_sha256", "pools_sha256", "index_sha256", "fitted_sha256", "models", "corpus_profile_partition")
    assert all(c["fixture_sha256"] == fixture_hash and all(c[k] == frozen[k] for k in keys) and
               c["gpu"] == "NVIDIA L4" and len(c["rows"]) == 48 for c in chunks)
    methods = set(chunks[0]["rows"][0]["order"])
    assert all(set(q["order"]) == set(q["records"]) == set(q["request_ms"]) == methods for c in chunks for q in c["rows"])
    for c in chunks:
        scope = {q["id"] for q in questions[c["shard"]::4][4:]}
        assert all(q["id"] in scope for q in c["rows"])
    rows = [r for c in chunks for r in c["rows"]]
    assert len(rows) == len({q["id"] for q in rows}) == 192
    methods = sorted(methods)
    assert {"exact", "full_rescore", "probe_64", "probe_256"} <= set(methods)
    out = {"fixture_sha256": fixture_hash, "partition_sha256": frozen["corpus_profile_partition"],
           "warm_index_seconds": [c["document_readiness_seconds"] for c in chunks],
           "all": summarize(rows, {q["id"]: q for q in questions}, methods)}
    a.output.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
