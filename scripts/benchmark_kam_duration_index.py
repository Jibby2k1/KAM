#!/usr/bin/env python3
"""Benchmark exact event lookup after swapping two durations in a serial plan."""

import argparse
from bisect import bisect_right
from hashlib import sha256
from itertools import accumulate
import json
from pathlib import Path
import platform
from random import Random
from statistics import median
from time import perf_counter_ns

import numpy as np

from audit_kam_unseentimeqa import DAY, clocks, question_key
from audit_kam_unseentimeqa_serial import ANCHOR_END, TARGET, serial_events


def replay(durations, anchor, offset, a, b):
    periods = durations.copy()
    periods[a], periods[b] = periods[b], periods[a]
    ends = list(accumulate(periods))
    point = offset + (ends[anchor - 1] if anchor else 0)
    return bisect_right(ends, point) if 0 <= point < ends[-1] else -1


def indexed(durations, ends, anchor, offset, a, b):
    delta = durations[b] - durations[a]
    point = offset + (ends[anchor - 1] if anchor else 0) + (delta if a < anchor <= b else 0)
    if not 0 <= point < ends[-1]:
        return -1
    if a == b:
        return bisect_right(ends, point)
    if a and point < ends[a - 1]:
        return bisect_right(ends, point, 0, a)
    if point < ends[b - 1] + delta:
        return bisect_right(ends, point - delta, a, b)
    return bisect_right(ends, point, b)


def batched(periods, ends, queries):
    anchor, offset, a, b = queries.T
    delta = periods[b] - periods[a]
    point = offset + np.where(anchor, ends[anchor - 1], 0) + delta * ((a < anchor) & (anchor <= b))
    before = (a > 0) & (point < ends[a - 1])
    during = (a < b) & (point < ends[b - 1] + delta)
    moved = np.clip(np.searchsorted(ends, point - delta, side="right"), a, b)
    fixed = np.searchsorted(ends, point, side="right")
    selected = np.where((a == b) | before, fixed, np.where(during, moved, np.maximum(fixed, b)))
    return np.where((point >= 0) & (point < ends[-1]), selected, -1)


def input_fingerprint(durations, queries):
    return sha256(np.asarray(durations, dtype="<i8").tobytes() +
                  np.asarray(queries, dtype="<i8").tobytes()).hexdigest()


def validate_authored(fixture):
    rows = json.loads(fixture.read_text())["rows"]
    h = sha256()
    for row in rows:
        source = row["variants"]["original"]
        events = serial_events(source["events"])
        durations = [e.end - e.start for e in events]
        ends = list(accumulate(durations))
        anchor = row["anchor"]
        clause, target = TARGET.split(row["question"], maxsplit=1)
        _, clock = ANCHOR_END.split(clause, maxsplit=1)
        [starting] = clocks(clock)
        source_minutes = (question_key(target)[1] - starting + events[anchor].start) % DAY
        offset = source_minutes - events[anchor].start
        options = [(anchor, offset, *(r["swap"] or [0, 0])) for r in row["variants"].values()]
        truth = [r["gold"] for r in row["variants"].values()]
        assert [replay(durations, *q) for q in options] == truth
        assert [indexed(durations, ends, *q) for q in options] == truth
        assert batched(np.array(durations), np.array(ends), np.array(options)).tolist() == truth
        h.update(input_fingerprint(durations, options).encode())
    return {"requests": 3 * len(rows), "fingerprint": h.hexdigest()}


def stress():
    rng = Random(519301)
    checked = 0
    for _ in range(60):
        durations = [rng.randint(1, 100) for _ in range(rng.randrange(3, 50))]
        ends = list(accumulate(durations))
        queries = []
        for _ in range(32):
            anchor = rng.randrange(len(durations))
            a, b = sorted(rng.sample(range(len(durations)), 2))
            shifted = list(accumulate(durations[:a] + [durations[b]] + durations[a+1:b] +
                                      [durations[a]] + durations[b+1:]))
            anchor_point = shifted[anchor - 1] if anchor else 0
            boundary = rng.choice([0, shifted[a], shifted[b - 1], shifted[-1], *shifted])
            queries.extend((anchor, boundary + d - anchor_point, a, b) for d in (-1, 0, 1))
        reference = [replay(durations, *q) for q in queries]
        assert [indexed(durations, ends, *q) for q in queries] == reference
        assert batched(np.array(durations), np.array(ends), np.array(queries)).tolist() == reference
        checked += len(queries)
    return checked


def benchmark(size, requests, repeats):
    rng = Random(4500 + size)
    durations = [rng.randrange(1, 100) for _ in range(size)]
    ends = list(accumulate(durations))
    queries = []
    for _ in range(requests):
        anchor = rng.randrange(size)
        a, b = sorted(rng.sample(range(size), 2))
        offset = rng.randrange(ends[-1]) - (ends[anchor - 1] if anchor else 0)
        queries.append((anchor, offset, a, b))
    arr, prefix, panel = np.array(durations), np.array(ends), np.array(queries)
    methods = {
        "rebuild": lambda: [replay(durations, *q) for q in queries],
        "indexed": lambda: [indexed(durations, ends, *q) for q in queries],
        "batched_indexed": lambda: batched(arr, prefix, panel).tolist(),
    }
    baseline = methods["rebuild"]()
    assert all(run() == baseline for run in methods.values())
    trials = {k: [] for k in methods}
    for _ in range(repeats):
        names = list(methods)
        rng.shuffle(names)
        for method in names:
            start = perf_counter_ns()
            result = methods[method]()
            trials[method].append((perf_counter_ns() - start) / requests)
            assert result == baseline
    cost = []
    for _ in range(repeats):
        start = perf_counter_ns()
        timings = list(accumulate(durations))
        cost.append(perf_counter_ns() - start)
        assert timings == ends
    return {"fingerprint": input_fingerprint(durations, queries), "requests": requests,
            "build_prefix_ns_p50": median(cost), "timing_ns_per_request": trials,
            "ns_per_request_p50": {method: median(values) for method, values in trials.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--repeats", default=8, type=int)
    p.add_argument("--requests", default=1024, type=int)
    a = p.parse_args()
    assert a.repeats > 0 and a.requests > 0
    result = {"machine": platform.machine(), "platform": platform.system(), "numpy": np.__version__,
              "fixture_sha256": sha256(a.fixture.read_bytes()).hexdigest(),
              "authored": validate_authored(a.fixture), "boundary_cases": stress(),
              "synthetic": {d: benchmark(d, a.requests, a.repeats) for d in (32, 256, 2048, 16384)}}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
