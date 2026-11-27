#!/usr/bin/env python3
"""Audit queue-inclusive latency and evidence decisions from matched arrivals."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import numpy as np

from analyze_kam_shared_serving import unpack
from kam_serving_arrivals import REQUESTS, arrivals

CAPS = {"fp16_width8": 8, "fp16_width16": 16, "int8_width16": 16}
RATES = (12, 24, 30)
SIZES = (1, 2, 4, 8, 16)


def quantiles(values):
    return np.quantile(values, [.5, .95]).tolist()


def checked_trial(trial, sample_ids, gold, seed):
    rate, name = trial["rate"], trial["policy"]
    cap = CAPS[name]
    assert trial["max_batch"] == cap and trial["max_wait_ms"] == 5 and trial["seed"] == seed
    expected = list(arrivals(rate, seed, sample_ids))
    durations = {k: [] for k in ("end_to_end_ms", "from_schedule_ms", "queue_ms", "arrival_jitter_ms", "service_ms")}
    decisions, busy, per_size = [], 0, Counter()
    last_finish, arrivals_seen = 0, []
    for batch in trial["batches"]:
        begin, end, count = batch["begin"], batch["finished"], len(batch["rows"])
        assert 0 <= last_finish <= begin < end and count in SIZES and count <= cap
        assert 0 < batch["service"] <= end - begin
        per_size[count] += 1
        last_finish = end
        busy += batch["service"]
        for row in batch["rows"]:
            seq = len(decisions)
            target, id_ = expected[seq]
            observed = row["arrival"]
            assert row["seq"] == seq and target == row["target"] and id_ == row["id"]
            assert target - .001 <= observed <= begin < end and len(row["selected_ids"]) == 2
            if arrivals_seen:
                assert observed >= arrivals_seen[-1]
            arrivals_seen.append(observed)
            for stage, duration in zip(durations, (end - observed, end - target, begin - observed,
                                                   observed - target, batch["service"]), strict=True):
                durations[stage].append(duration * 1000)
            exact = bool(row["answer_exact"])
            ordered = tuple(row["selected_ids"])
            support = set(ordered) == gold[id_]
            decisions.append((id_, ordered, support, exact, support and exact))
    assert len(decisions) == REQUESTS
    period = last_finish - arrivals_seen[0]
    quality = dict(zip(("support", "answer", "joint"), map(sum, zip(*(r[2:] for r in decisions))), strict=True))
    stats = {key: quantiles(v) for key, v in durations.items()}
    stats.update(offered_qps=(REQUESTS-1)/(arrivals_seen[-1]-arrivals_seen[0]), completed_qps=REQUESTS/period,
                 drain_ms=(last_finish-arrivals_seen[-1])*1000, occupancy=busy/period, widths=per_size, counts=quality)
    return stats, decisions


def report(measurement, reference, gold, seal):
    all_trials = {(r, arm): [] for r in RATES for arm in CAPS}
    agreement = {key: Counter() for key in all_trials}
    trace_counts = {}
    for i, (raw, past) in enumerate(zip(measurement, reference, strict=True)):
        assert raw["fixture_sha256"] == past["fixture_sha256"] == seal
        assert raw["gpu"] == past["gpu"] == "NVIDIA L4"
        assert raw["shards"] == past["shards"] == 4 and raw["shard"] == past["shard"] == i
        assert all(raw[k] == past[k] for k in ("index_sha256", "fitted_sha256", "models", "corpus_profile_partition"))
        sample_ids = [id_ for b in past["rows"] for id_ in b["ids"]]
        assert len(sample_ids) == len(set(sample_ids)) == 200
        assert {(t["rate"], t["policy"]) for t in raw["rows"]} == all_trials.keys()
        prior = {}
        for trial in raw["rows"]:
            k = trial["rate"], trial["policy"]
            stats, decisions = checked_trial(trial, sample_ids, gold, 829559 + i)
            all_trials[k].append(stats)
            prior[k] = decisions
        for k, observations in prior.items():
            ref = prior[k[0], "fp16_width8"]
            for old, new in zip(ref, observations, strict=True):
                assert old[0] == new[0]
                agreement[k].update(pair=old[1] == new[1], support=old[2] == new[2],
                                    answer=old[3] == new[3], joint=old[4] == new[4],
                                    joint_wins=new[4] and not old[4], joint_losses=old[4] and not new[4])
            trace_counts.setdefault(k[0], set()).update(r[0] for r in observations)
    result = {}
    for (rate, policy), runs in all_trials.items():
        result.setdefault(str(rate), {})[policy] = {
            "per_shard": runs, "unique_questions": len(trace_counts[rate]), "parity_vs_fp16_width8": agreement[rate, policy],
            "median_shard_completed_qps": median(r["completed_qps"] for r in runs),
            "median_shard_offered_qps": median(r["offered_qps"] for r in runs),
            "median_shard_p95_ms": {k: median(r[k][1] for r in runs) for k in
                                    ("end_to_end_ms", "from_schedule_ms", "queue_ms", "arrival_jitter_ms")},
            "median_shard_drain_ms": median(r["drain_ms"] for r in runs)}
    return {"fixture_sha256": seal, "device": "NVIDIA L4", "trials": result, "requests_per_policy_rate": 4 * REQUESTS}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("measurement", "reference"):
        p.add_argument("--" + k, type=Path, nargs=4, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fixture = gzip.decompress(a.fixture.read_bytes())
    gold = {q["id"]: set(q["gold_ids"]) for q in json.loads(fixture)["rows"]["test"]}
    measurement, reference = [sorted(map(unpack, v), key=lambda r: r["shard"]) for v in (a.measurement, a.reference)]
    result = report(measurement, reference, gold, sha256(fixture).hexdigest())
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
