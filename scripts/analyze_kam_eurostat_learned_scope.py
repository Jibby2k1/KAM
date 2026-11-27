#!/usr/bin/env python3
"""Verify paired comparisons across archived Eurostat scope policies."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_shared_serving import unpack


def audit(paths):
    all_results = {}
    inputs = {}
    for path in paths:
        data = unpack(path)
        key = data.get("mixed_ambiguous_cues", False), data["cue_only"]
        assert key not in all_results
        all_results[key] = data
        inputs[f"{'mixed' if key[0] else 'mirrored'}/{'cue' if key[1] else 'raw'}"] = sha256(path.read_bytes()).hexdigest()
    expected = all_results[False, False]
    assert len(all_results) == 4 and expected["cue_families"] == {"fit": 4, "calibration": 2, "test": 6}
    assert expected["split_sizes"] == {"fit": 432, "calibration": 432, "test": 1296}
    published = {}
    for (mixed, cue), data in all_results.items():
        for key in ("fixture_sha256", "ranked_sha256", "device", "split_sizes", "cue_families", "reader_sha256", "reader_revision"):
            assert data[key] == expected[key]
        families = defaultdict(Counter)
        totals = defaultdict(Counter)
        for row, reference in zip(data["rows"], expected["rows"], strict=True):
            assert {k: v for k, v in row.items() if k != "observed"} == {k: v for k, v in reference.items() if k != "observed"}
            kind, target = row["kind"], row["target"]
            assert kind in {"single", "neither", "both"} and ((kind == "single") == (target < 2))
            assert row["observed"]["exact_scope"] == {"choice": target, "release": row["target_release"]}
            families[(kind, row["family"])]["n"] += 1
            totals[kind]["n"] += 1
            for policy, result in row["observed"].items():
                correct = result == row["observed"]["exact_scope"]
                assert correct == (result["choice"] == target)
                if kind == "single" and not cue and policy.endswith("_forced"):
                    assert result == reference["observed"][policy]
                totals[kind][policy] += correct
                families[(kind, row["family"])][policy] += correct
        for kind, stats in totals.items():
            assert data["outcomes"][kind]["n"] == stats["n"]
            assert all(data["outcomes"][kind]["policies"][p]["correct_roles"] == n for p, n in stats.items() if p != "n")
        name = f"{'mixed' if mixed else 'mirrored'}/{'cue' if cue else 'raw'}"
        published[name] = {"thresholds": data["calibration"], "cohorts": data["outcomes"],
                           "cue_family_counts": {f"{kind}/{family}": counts for (kind, family), counts in families.items()}}
    return {"fixture_sha256": expected["fixture_sha256"], "ranked_sha256": expected["ranked_sha256"],
            "inputs_sha256": inputs, "cohorts": published}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(audit(args.inputs), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
