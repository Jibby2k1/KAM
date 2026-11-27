#!/usr/bin/env python3
"""Summarize exact decision checks and synchronized suffix-compaction timing."""

import argparse
import json
from pathlib import Path
from statistics import median

from analyze_kam_captured_wallclock import read

METHODS = ("full", "batch_max", "length_buckets")


def aggregate(files):
    runs = sorted(map(read, files), key=lambda q: q["seed"])
    assert len({p["seed"] for p in runs}) == len(runs) == 3
    assert len({p["device"] for p in runs}) == len({p["checkpoint_sha256"] for p in runs}) == 1
    out = {}
    for name in runs[0]["panels"]:
        panel = [r["panels"][name] for r in runs]
        assert all(p["shape"] == panel[0]["shape"] for p in panel)
        for label in panel[0]["profiles"]:
            compact = [p["profiles"][label] for p in panel]
            tested = sum(p["shape"][0] for p in panel)
            profile = {"tested": tested, "tokens": panel[0]["shape"][1],
                       "eligible_batch_sizes": [c["timed_eligible"] for c in compact], "modes": {}}
            for m in METHODS:
                checks = [d["parity"][m] for d in compact]
                times = [d["synchronized_timings"][m] for d in compact]
                base = [d["synchronized_timings"]["full"] for d in compact]
                profile["modes"][m] = {"matching_decisions": sum(r["match"] for r in checks),
                                       "correct_decisions": sum(r["joint"] for r in checks),
                                       "largest_logit_delta": max(r["max_delta"] for r in checks),
                                       "wall_ms": median(r["wall_ms"] for r in times),
                                       "cuda_stream_ms": median(r["stream_ms"] for r in times),
                                       "additional_peak_bytes": median(r["extra_peak_bytes"] for r in times),
                                       "seed_speedup": [u["wall_ms"]/v["wall_ms"] for u, v in zip(base, times, strict=True)]}
            out[f"{name}/{label}"] = profile
    return {"device": runs[0]["device"], "checkpoint_sha256": runs[0]["checkpoint_sha256"], "panels": out,
            "input_hashes": {str(r["seed"]): {k: v["fixture_sha256"] for k, v in r["panels"].items()} for r in runs}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l4", required=True, nargs=3, type=Path)
    p.add_argument("--b200", nargs=3, type=Path)
    p.add_argument("--rtx6000", nargs=3, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = {"l4": aggregate(a.l4)}
    for device, files in (("b200", a.b200), ("rtx6000", a.rtx6000)):
        if files:
            result[device] = aggregate(files)
            assert result["l4"]["input_hashes"].keys() == result[device]["input_hashes"].keys()
            assert ("B200" if device == "b200" else "RTX PRO 6000") in result[device]["device"]
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
