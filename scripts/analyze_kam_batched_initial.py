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
STAGES = ("request_ms", "initial_search_ms", "reader_ms")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in ("measurement", "reference"):
        p.add_argument("--" + name, type=Path, nargs=4, required=True)
    p.add_argument("--device", choices=("NVIDIA L4", "NVIDIA B200"), default="NVIDIA L4")
    a = p.parse_args()
    raw = gzip.decompress(a.fixture.read_bytes())
    fx = sha256(raw).hexdigest()
    truth = {x["id"]: set(x["gold_ids"]) for x in json.loads(raw)["rows"]["test"]}
    experiments, replay = [sorted(map(unpack, args), key=lambda c: c["shard"]) for args in (a.measurement, a.reference)]
    times = {m: {phase: [] for phase in STAGES} for m in ARMS}
    flags = {m: {k: [] for k in ("support", "answer", "joint")} for m in ARMS}
    parity = Counter()
    distinct = set()
    for i, (done, baseline) in enumerate(zip(experiments, replay, strict=True)):
        assert done["fixture_sha256"] == baseline["fixture_sha256"] == fx
        assert done["shard"] == baseline["shard"] == i and done["shards"] == baseline["shards"] == 4
        assert all(done[k] == baseline[k] for k in ("models", "index_sha256", "fitted_sha256", "corpus_profile_partition"))
        assert done["gpu"] == a.device and baseline["gpu"] == "NVIDIA L4"
        assert len(done["rows"]) == len(baseline["rows"]) == 25 and set(done["batch_policies"]) == set(ARMS)
        for trial, anchor in zip(done["rows"], baseline["rows"], strict=True):
            assert trial["ids"] == anchor["ids"] and len(trial["ids"]) == 8
            assert set(trial["order"]) == set(trial["batches"]) == set(ARMS)
            assert distinct.isdisjoint(trial["ids"])
            distinct.update(trial["ids"])
            separate, shared = [trial["batches"][m] for m in ARMS]
            for name, stage in zip(ARMS, (separate, shared), strict=True):
                timing, = stage["timing"]
                assert timing["size"] == 8
                for phase in STAGES:
                    times[name][phase].append(timing[phase])
            before = anchor["batches"]["fp16/8"]["rows"]
            for ref, one, many, id_ in zip(before, separate["rows"], shared["rows"], trial["ids"], strict=True):
                assert ref["id"] == one["id"] == many["id"] == id_
                selected = [r["arms"]["conditioned_union"]["pair"] for r in (one, many)]
                parity.update({"separate_vs_archive": one == ref, "arms_vs_archive": one["arms"] == ref["arms"],
                               "initial_set": set(one["initial_ids"]) == set(many["initial_ids"]),
                               "initial_order": one["initial_ids"] == many["initial_ids"],
                               "second_hop": one["corpus_hits"] == many["corpus_hits"],
                               "conditioned_pairs": selected[0] == selected[1], "all_records": one == many})
                for name, out, pair in zip(ARMS, (one, many), selected, strict=True):
                    support = set(pair) == truth[id_]
                    answer = out["arms"]["conditioned_union"]["exact"]
                    for key, hit in zip(flags[name], (support, answer, support and answer), strict=True):
                        flags[name][key].append(hit)
    n = len(distinct)
    assert n == 800
    sample = np.random.default_rng(720528).integers(0, 100, size=(4000, 100))
    summaries = {}
    for phase in STAGES:
        old, new = [np.asarray(times[arm][phase]) for arm in ARMS]
        assert len(old) == len(new) == 100
        delta = new - old
        summaries[phase] = {"p50_ms": dict(zip(ARMS, np.median([old, new], axis=1).tolist(), strict=True)),
                            "p95_ms": dict(zip(ARMS, np.quantile([old, new], .95, axis=1).tolist(), strict=True)),
                            "paired_median_delta_ms": float(np.median(delta)),
                            "paired_median_delta_95_ms": np.quantile(np.median(delta[sample], axis=1), [.025, .975]).tolist()}
    result = {"fixture_sha256": fx, "device": a.device, "requests": n, "parity": parity, "latency": summaries,
              "requests_per_second": {m: 1000*n/sum(times[m]["request_ms"]) for m in ARMS},
              "counts": {m: {k: sum(row) for k, row in f.items()} for m, f in flags.items()},
              "quality": {k: compare(flags["shared"][k], v, [[i] for i in range(n)]) for k, v in flags["separate"].items()}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
