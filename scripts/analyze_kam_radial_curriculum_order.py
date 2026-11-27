#!/usr/bin/env python3
"""Audit matched data order controls and score radial learning trajectories."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from analyze_kam_sentence_verifier import paired

ARMS = ("interleaved", "short_first")
METRICS = ("entity", "record", "joint")


def cases(sample):
    return {key: [int(a if key == "entity" else b if key == "record" else a and b)
                  for a, b in sample["outcomes"]] for key in METRICS}


def run(record):
    reference, staged = [record["policies"][a]["held_out"] for a in ARMS]
    assert reference.keys() == staged.keys() and reference["64"]["outcomes"] == staged["64"]["outcomes"]
    final = str(record["steps"])
    assert reference[final]["tokens"] == staged[final]["tokens"]
    assert reference[final]["depth_updates"] == staged[final]["depth_updates"]
    assert all(x == record["steps"] // len(record["config"]) for x in reference[final]["depth_updates"].values())
    out = {}
    for step in sorted(reference, key=int):
        x, y = reference[step], staged[step]
        data = {}
        assert set(x["outcomes"]) == set(y["outcomes"]) == set(record["initial_validation"])
        for depth, sx in x["outcomes"].items():
            sy = y["outcomes"][depth]
            assert sx["fixture_sha256"] == sy["fixture_sha256"] == record["initial_validation"][depth]["fixture_sha256"]
            xcase, ycase = cases(sx), cases(sy)
            data[depth] = {m: {"reference_mean": mean(xcase[m]), "short_first_mean": mean(ycase[m]),
                                "delta": paired(ycase[m], xcase[m])} for m in METRICS}
        out[step] = {"outcomes": data,
                     "tokens": {a: r["tokens"] for a, r in zip(ARMS, (x, y), strict=True)},
                     "elapsed_update_seconds": {a: r["elapsed_update_seconds"] for a, r in zip(ARMS, (x, y), strict=True)},
                     "depth_updates": {a: r["depth_updates"] for a, r in zip(ARMS, (x, y), strict=True)}}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs=3, type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    items = sorted((json.loads(p.read_text()) for p in a.runs), key=lambda r: r["seed"])
    assert len({x["seed"] for x in items}) == 3
    assert all(r["steps"] == items[0]["steps"] and r["device"] == items[0]["device"]
               and r["config"] == items[0]["config"] for r in items)
    out = {k: items[0][k] for k in ("steps", "config", "device")}
    out["seeds"] = {r["seed"]: run(r) for r in items}
    out["input_sha256"] = sorted(sha256(p.read_bytes()).hexdigest() for p in a.runs)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
