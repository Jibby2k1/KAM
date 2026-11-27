#!/usr/bin/env python3
"""Validate exact packed date sorting and summarize local speedups by device."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from analyze_kam_guarded_fp32_sdpa import load


def audit(records):
    grouped, fingerprints = defaultdict(list), {}
    for run in records:
        grouped[run["device"]].append(run)
        seal = run["weights_sha256"], {p: q["fixture_sha256"] for p, q in run["panels"].items()}
        assert fingerprints.setdefault(run["seed"], seal) == seal
        assert run["archive_inputs_sha256"] == records[0]["archive_inputs_sha256"]
    results = {}
    for device, trials in grouped.items():
        assert len(trials) == len({r["seed"] for r in trials}) == 3
        grids = {}
        for name in trials[0]["panels"]:
            panels = [r["panels"][name] for r in trials]
            for group in panels:
                assert all(k["tested"] == k["exact_readout_choices"] == 64 for k in group["parity"].values())
                assert group["input_metadata_bytes"] == panels[0]["input_metadata_bytes"]
            estimates = {k: median(p["times"][k]["wall_ms"] for p in panels) for k in panels[0]["times"]}
            changes = {}
            for g in ("dot", "radial"):
                base, packed = f"{g}/reference", f"{g}/packed"
                pairs = [p["times"][base]["wall_ms"] - p["times"][packed]["wall_ms"] for p in panels]
                changes[g] = {"median_difference_ms": median(pairs), "per_seed_differences_ms": pairs,
                              "layout": panels[0]["parity"][g]["layout"]}
            grids[name] = {"wall_ms": estimates, "paired_fused_reductions": changes,
                           "metadata_bytes_at_timed_size": panels[0]["input_metadata_bytes"], "batch": panels[0]["batch"]}
        assert all(p["input_parity_archive_counts"] == trials[0]["input_parity_archive_counts"] for p in trials)
        results[device] = {"archive_input_tensor_parity": {k: sum(p["input_parity_archive_counts"][k] for p in trials)
                                                           for k in trials[0]["input_parity_archive_counts"]},
                           "panels": grids}
    return {"devices": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(audit(list(map(load, a.measurements))), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
