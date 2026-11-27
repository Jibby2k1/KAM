#!/usr/bin/env python3
"""Replay a mean-duration control with access to the exact suffix clock span."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import torch

from analyze_kam_duration_coordinates import SEEDS, SHAPES
from analyze_kam_shared_serving import unpack
from run_kam_duration_pointer import prepared_fixture, sample, sha


def suffix_mean(duration, anchor, offset):
    n = duration.shape[1]
    in_suffix = torch.arange(n)[None] >= anchor[:, None]
    span = (duration * in_suffix).sum(-1)
    return anchor + offset * (n - anchor) // span


def replay(raw, fixture, runs):
    synthetic = {}
    for i, shape in enumerate(SHAPES):
        n, maximum = map(int, shape.split("/"))
        synthetic[shape] = {}
        for seed in SEEDS:
            panel = sample(seed + 612991 + i, n, 512, maximum)
            assert sha(panel) == runs[seed]["synthetic"][shape]["sha256"]
            d, a, t, target = panel
            synthetic[shape][str(seed)] = int(suffix_mean(d, a, t).eq(target).sum())
    archived, by_case = defaultdict(int), defaultdict(list)
    for entries in prepared_fixture(fixture["rows"]).values():
        d, a, t, target = (torch.tensor(col) for col in zip(*(r[2:] for r in entries), strict=True))
        predictions = suffix_mean(d, a, t)
        for (case, scenario, *_), selection, correct in zip(entries, predictions.tolist(), predictions.eq(target).tolist(),
                                                            strict=True):
            archived[scenario] += int(correct)
            by_case[case].append(selection)
    assert all(len(z) == 3 and len(set(z)) == 1 for z in by_case.values())
    return {"fixture_sha256": sha256(raw).hexdigest(), "synthetic_correct_of_512": synthetic,
            "archived_correct_of_1173": dict(archived) | {"both_intervention_and_original": 0}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs=3, type=Path)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    raw = args.fixture.read_bytes()
    runs = {run["seed"]: run for run in map(unpack, args.runs)}
    assert set(runs) == set(SEEDS) and all(r["fixture_sha256"] == sha256(raw).hexdigest() for r in runs.values())
    result = replay(raw, json.loads(raw), runs)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
