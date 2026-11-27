#!/usr/bin/env python3
"""Audit exact duration-index timing inputs and recompute speed ratios."""

import argparse
import json
from pathlib import Path
from statistics import median


def summarize(local, hpg):
    assert local["platform"] == "Darwin" and local["machine"] == "arm64"
    assert hpg["platform"] == "Linux" and hpg["machine"] == "x86_64"
    assert local["fixture_sha256"] == hpg["fixture_sha256"]
    assert local["authored"] == hpg["authored"] and local["authored"]["requests"] == 3519
    assert local["boundary_cases"] == hpg["boundary_cases"] == 5760
    sizes = ("32", "256", "2048", "16384")
    methods = {"rebuild", "indexed", "batched_indexed"}
    assert set(local["synthetic"]) == set(hpg["synthetic"]) == set(sizes)
    assert all(local["synthetic"][s]["fingerprint"] == hpg["synthetic"][s]["fingerprint"] for s in sizes)
    hosts = {}
    for key, record in (("apple", local), ("hpg", hpg)):
        evidence = {}
        for size in sizes:
            measured = record["synthetic"][size]
            assert measured["requests"] == 1024 and measured["build_prefix_ns_p50"] > 0
            timing = measured["timing_ns_per_request"]
            assert set(timing) == methods and all(len(x) == 8 and min(x) > 0 for x in timing.values())
            middle = {m: median(v) for m, v in timing.items()}
            assert middle == measured["ns_per_request_p50"]
            evidence[size] = {"prefix_us": round(measured["build_prefix_ns_p50"] / 1_000, 6),
                              "p50_us_per_request": {m: round(v / 1_000, 6) for m, v in middle.items()},
                              "rebuild_speed_ratio": {m: round(middle["rebuild"] / middle[m], 6)
                                                      for m in ("indexed", "batched_indexed")}}
        hosts[key] = {"numpy": record["numpy"], "measurements": evidence}
    return {"fixture_sha256": local["fixture_sha256"], "authored": local["authored"],
            "boundary_cases": local["boundary_cases"], "hosts": hosts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("apple", "hpg", "output"):
        p.add_argument("--" + arg, required=True, type=Path)
    args = p.parse_args()
    result = summarize(json.loads(args.apple.read_text()), json.loads(args.hpg.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
