#!/usr/bin/env python3
"""Audit token-matched depth exposure on procedural and archived lookup."""

import argparse
import json
from pathlib import Path
from statistics import mean

from analyze_kam_archive_clock_encoding import split_inputs
from analyze_kam_external_clock_transfer import indexed, MODES, PANELS
from analyze_kam_native_musique import compare
from analyze_kam_oecd_revisions import paired_quarter_seed
from analyze_kam_temporal_clock_encoding import GEOMETRIES

DOMAINS = ("corrections", "mixed")
LENGTHS = ("16", "48", "128", "512", "1024")


def examine(short, longer, geometry, encoding):
    key = f"{geometry}/{encoding}"
    reports = {}
    for kind in ("original", "corrections"):
        data = {}
        for d in LENGTHS:
            comparisons = {}
            for seed in sorted(short):
                old, new = [p[seed]["results"][key]["procedural"][kind][d] for p in (short, longer)]
                assert old["fixture_sha256"] == new["fixture_sha256"]
                a, b = [list(map(all, r["outcomes"])) for r in (old, new)]
                comparisons[str(seed)] = {"n": len(a), "short_hits": sum(a), "long_hits": sum(b),
                                           "paired": compare(b, a, [[i] for i in range(len(a))])}
            data[d] = {"short_mean": mean(r["short_hits"]/r["n"] for r in comparisons.values()),
                       "long_mean": mean(r["long_hits"]/r["n"] for r in comparisons.values()), "seeds": comparisons}
        reports[kind] = data
    archive, external = {}, {}
    for p in ("earlier", "later"):
        archive[p] = {}
        for depth in (16, 48):
            protocol = [indexed(r, geometry, encoding, p, depth) for r in (short, longer)]
            a, b = protocol
            assert a.keys() == b.keys()
            archive[p][str(depth)] = {"n": len(a), "short_hits": sum(q["joint"] for q in a.values()),
                                      "long_hits": sum(q["joint"] for q in b.values()),
                                      "paired": paired_quarter_seed({k: int(b[k]["joint"]) - int(a[k]["joint"]) for k in a})}
    for p in PANELS:
        external[p] = {}
        for depth in (16, 48):
            a, b = [indexed(r, geometry, encoding, p, depth) for r in (short, longer)]
            assert a.keys() == b.keys() and len(a) == 3504
            external[p][str(depth)] = {"n": len(a), "short_hits": sum(r["joint"] for r in a.values()),
                                       "long_hits": sum(r["joint"] for r in b.values()),
                                       "paired": paired_quarter_seed({k: int(b[k]["joint"]) - int(a[k]["joint"]) for k in a})}
    return {"procedural": reports, "philadelphia": archive, "dallas": external}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for domain in DOMAINS:
        for curriculum in ("short", "long"):
            p.add_argument(f"--{domain}-{curriculum}", required=True, nargs=3, type=Path)
    p.add_argument("--mixed-long-delayed", nargs=3, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    summary = {}
    for name in DOMAINS:
        short, long = [split_inputs(getattr(a, f"{name}_{m}"), name) for m in ("short", "long")]
        assert short.keys() == long.keys()
        for seed, start in short.items():
            next_ = long[seed]
            assert start["train_tokens"] == next_["train_tokens"] == 10556800
            assert start["archive_sha256"] == next_["archive_sha256"] and start["steps"] == next_["steps"] == 3200
            assert [r[0] for r in start["schedule"]] == [16, 48]
            assert [r[0] for r in next_["schedule"]] == [16, 48, 128, 512]
            assert all(start["banks"][d]["scene_sha256"] == next_["banks"][d]["scene_sha256"] for d in ("16", "48"))
            assert {mode: info["initial_sha256"] for mode, info in start["results"].items()} == {
                mode: info["initial_sha256"] for mode, info in next_["results"].items()}
        summary[name] = {f"{g}/{mode}": examine(short, long, g, mode) for g in GEOMETRIES for mode in MODES}
    result = {"steps": 3200, "tokens": 10556800, "domains": summary}
    if a.mixed_long_delayed:
        reference, delayed = [split_inputs(inputs, "mixed") for inputs in (a.mixed_long, a.mixed_long_delayed)]
        assert reference.keys() == delayed.keys()
        for seed, r in reference.items():
            d = delayed[seed]
            assert all(r[k] == d[k] for k in ("device", "steps", "schedule", "train_tokens", "archive_sha256", "banks"))
            assert d["short_first_prefix_updates"] == 800 and len(d["order_sha256"]) == 64
            assert {mode: info["initial_sha256"] for mode, info in r["results"].items()} == {
                mode: info["initial_sha256"] for mode, info in d["results"].items()}
        result["mixed_delayed_vs_cyclic"] = {
            f"{g}/{mode}": examine(reference, delayed, g, mode) for g in GEOMETRIES for mode in MODES}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
