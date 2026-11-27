#!/usr/bin/env python3
"""Audit paired optimizer capture and long-history lookup outcomes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from analyze_kam_guarded_fp32_sdpa import load


def paired(before, after):
    flags = list(zip(before, after, strict=True))
    return {"tested": len(flags), "correct": sum(after),
            "wins": sum(b and not a for a, b in flags), "losses": sum(a and not b for a, b in flags)}


def outcomes(rows, label, reference):
    known = [r["arms"][reference] for r in rows]
    varied = [r["arms"][label] for r in rows]
    rr = rows[0]["arms"][reference]
    synthetic, archives = {}, {}
    for kind, panels in rr["heldout"].items():
        for d in panels:
            actual, baseline = [], []
            for a, b in zip(known, varied, strict=True):
                p, q = (data["heldout"][kind][d] for data in (a, b))
                assert p["fixture_sha256"] == q["fixture_sha256"]
                baseline.extend(p["outcomes"])
                actual.extend(q["outcomes"])
            synthetic[f"{kind}/{d}"] = paired(list(map(all, baseline)), list(map(all, actual))) | {
                "alias_correct": sum(x[0] for x in actual), "snapshot_correct": sum(x[1] for x in actual)}
    for topic in rr["archives"]:
        records = defaultdict(list)
        for a, b in zip(known, varied, strict=True):
            for p, q in zip(a["archives"][topic], b["archives"][topic], strict=True):
                assert all(p[k] == q[k] for k in ("depth", "recode", "fixture_sha256"))
                for x, y in zip(p["outcomes"], q["outcomes"], strict=True):
                    assert (x["id"], x["period"], x["series"]) == (y["id"], y["period"], y["series"])
                    records[p["depth"]].append((x, y))
        for depth, data in records.items():
            archives[f"{topic}/{depth}"] = paired([x["joint"] for x, _ in data], [y["joint"] for _, y in data]) | {
                "numeric_correct": sum(y["numeric_exact"] for _, y in data)}
    return {"median_seconds_with_setup": median(x["setup_seconds"]+x["wall_seconds"] for x in varied),
            "median_training_loop_seconds": median(x["wall_seconds"] for x in varied),
            "median_extra_allocated_bytes": median(x["peak_extra_allocated_bytes"] for x in varied),
            "final_losses": [x["loss"] for x in varied], "synthetic": synthetic, "archives": archives}


def audit(records, order_only):
    groups = defaultdict(list)
    reference = "interleaved" if order_only else "eager_attention/eager"
    for row in records:
        assert row["steps"] == 3200 and row["schedule"] == [[16, 82], [48, 42], [128, 13], [512, 3]]
        assert row["archives_sha256"] == records[0]["archives_sha256"]
        groups[row["device"]].append(row)
    assert len(groups) == (1 if order_only else 2)
    results = {}
    for device, rows in groups.items():
        rows.sort(key=lambda e: e["seed"])
        assert len(rows) == len({x["seed"] for x in rows}) == 3
        output = {}
        for g in ("dot", "radial"):
            values = [x["results"][g] for x in rows]
            if order_only:
                assert all(x["interleaved_archive_replay"] for x in values)
                assert all(len(set(r["sequence_sha256"].values())) == 3 for r in rows)
                flags = {"archived_interleaved_replay": True}
            else:
                flags = {}
                for k in ("eager_attention", "fused_fp32"):
                    parity = [x["capture_parity"][k] for x in values]
                    assert all(y["weights"]["equal"] and y["moments"]["equal"] for y in parity)
                    assert all(y["same_final_loss"] and y["same_heldout"] and y["same_archive"] for y in parity)
                    flags[k] = parity
                flags["weight_difference_across_kernels"] = [x["weights_eager_vs_fused"] for x in values]
            output[g] = {"outcomes": {arm: outcomes(values, arm, reference) for arm in values[0]["arms"]},
                         "verification": flags}
        results[device] = output
    return {"paired_input_tokens": 10556800, "reference": reference, "devices": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, nargs="+", required=True)
    p.add_argument("--order-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(list(map(load, args.measurements)), args.order_only)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
