#!/usr/bin/env python3
"""Audit FP32 decision fidelity and fallback cost on separate measurement histories."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import median

POLICIES = ("eager", "flash", "precise_readout", "calibrated_fallback")


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def measure(panels):
    output = {}
    for mode in POLICIES:
        stats = Counter()
        timing = []
        for r in panels:
            flags = r[mode]
            alias, fact, match = (flags[k] for k in ("correct_keys", "correct_values", "full_match"))
            n = len(match)
            assert len(alias) == len(fact) == n
            stats.update({"observations": n, "alias": sum(alias), "fact": sum(fact),
                          "joint": sum(a and b for a, b in zip(alias, fact, strict=True)),
                          "decision_matches": sum(match)})
            if mode == "eager":
                assert all(match)
            if mode == "calibrated_fallback":
                requests = flags["fallback"]
                assert len(requests) == n
                fp32_mismatch = [i for i, v in enumerate(match) if not v]
                flash_mismatch = [i for i, v in enumerate(r["flash"]["full_match"]) if not v]
                assert all(requests[i] or not r["flash"]["full_match"][i] for i in fp32_mismatch)
                stats.update({"fallback_requests": sum(requests),
                              "unflagged_disagreements": sum(not requests[i] for i in flash_mismatch),
                              "flagged_still_disagreed": sum(requests[i] for i in fp32_mismatch)})
            if "timing" in flags:
                timing.append(flags["timing"])
        output[mode] = dict(stats) | {"largest_logit_error": [
            max(r[mode]["max_logit_differences"][i] for r in panels) for i in (0, 1)]}
        if timing:
            output[mode]["timing_seed_medians"] = timing
            output[mode]["median_wall_ms"] = median(r["wall_ms"] for r in timing)
            output[mode]["median_transient_bytes"] = median(r["extra_peak_bytes"] for r in timing)
    return output


def run(inputs):
    assert len(inputs) == len({r["seed"] for r in inputs}) == 3
    assert {r["device"] for r in inputs} == {"NVIDIA L4"}
    assert all(r["clock"] == inputs[0]["clock"] and r["archives_sha256"] == inputs[0]["archives_sha256"] for r in inputs)
    for run in inputs:
        evaluation = {p["fixture_sha256"] for p in run["procedural"].values()}
        calibration = {h for d in run["calibration"].values() for h in d["fixture_hashes"]}
        assert evaluation.isdisjoint(calibration)
        assert all(p["tests"] == 192 for p in run["calibration"].values())
    procedural = {k: measure([r["procedural"][k]["test"] for r in inputs]) for k in inputs[0]["procedural"]}
    archives = {}
    for tag, cases in inputs[0]["archive"].items():
        depths = sorted({r["depth"] for r in cases})
        archives[tag] = {d: measure([panel["test"] for trial in inputs for panel in trial["archive"][tag]
                                    if panel["depth"] == d]) for d in depths}
    return {"device": inputs[0]["device"], "clock": inputs[0]["clock"],
            "archives_sha256": inputs[0]["archives_sha256"],
            "calibration": {r["seed"]: r["calibration"] for r in inputs}, "procedural": procedural, "archives": archives}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run(list(map(read, a.measurement)))
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
