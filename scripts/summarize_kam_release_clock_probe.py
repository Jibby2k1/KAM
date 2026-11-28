#!/usr/bin/env python3
"""Replay live release-clock choices and isolate eligibility and freshness errors."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, digest
from kam_math_kl_protocol import write_json
from kam_release_clock_probe import CLOCKS, LETTERS, NONE, PROTOCOL, latest
from summarize_kam_overnight_routes import verified
from summarize_kam_pretrained_endpoint_selection import family_difference

MODES = ("direct", "restricted")


def summarize(work, manifest, panel, study):
    fixed = json.loads(manifest.read_text())
    cases = json.loads(panel.read_text())
    assert fixed["protocol"] == PROTOCOL and fixed["panel_sha256"] == digest(panel)
    assert fixed["cases"] == [{k: r[k] for k in ("id", "category", "project")} for r in cases]
    checkpoints, profiles, hashes = {}, defaultdict(dict), {}
    for name, run, data, fingerprint in verified(work, manifest, study, "clock", PROTOCOL):
        rows = iter(data["records"])
        hashes[name], checkpoints[name] = fingerprint, defaultdict(dict)
        category_modes = defaultdict(lambda: defaultdict(list))
        for case in cases:
            category = case["category"]
            metrics = checkpoints[name][category]
            version = [r["version"] for r in case["records"]] + [NONE]
            by_mode = {mode: [] for mode in MODES}
            for clock in CLOCKS:
                goal = latest(case["records"], clock, case["cutoff"])
                assert goal == case["gold"][clock]
                eligible = [r["version"] for r in case["records"] if r[clock] <= case["cutoff"]] or [NONE]
                other = case["gold"][CLOCKS[1 - CLOCKS.index(clock)]]
                for shift in range(len(LETTERS)):
                    doc = next(rows)
                    slots = version[shift:] + version[:shift]
                    expected = (case["id"], case["project"], category, clock, shift)
                    assert tuple(doc[key] for key in ("id", "project", "category", "clock", "offset")) == expected
                    assert doc["gold"] == slots.index(goal)
                    allowed = [j for j, v in enumerate(slots) if v in eligible]
                    assert doc["eligible"] == allowed
                    logits = doc["logits"]
                    assert 0 < doc["tokens"] and len(logits) == 4 and np.isfinite(logits).all()
                    assert doc["direct"] == max(range(4), key=logits.__getitem__)
                    assert doc["restricted"] == max(allowed, key=logits.__getitem__)
                    direct = slots[doc["direct"]]
                    metrics["presentations"] = metrics.get("presentations", 0) + 1
                    for label, flag in (
                        ("ineligible_choice", direct not in eligible),
                        ("stale_eligible_choice", direct in eligible and direct != goal),
                        ("other_clock_choice", goal != other and direct == other),
                        ("multiple_eligible", len(eligible) > 1),
                        ("multiple_eligible_direct_correct", len(eligible) > 1 and direct == goal),
                        ("multiple_eligible_restricted_correct", len(eligible) > 1 and doc["restricted"] == doc["gold"]),
                    ):
                        metrics[label] = metrics.get(label, 0) + int(flag)
                    for mode in MODES:
                        by_mode[mode].append(doc[mode] == doc["gold"])
            for mode, flags in by_mode.items():
                metrics[f"{mode}_correct"] = metrics.get(f"{mode}_correct", 0) + sum(flags)
                category_modes[category][mode].append(all(flags))
        assert next(rows, None) is None
        for category, columns in category_modes.items():
            for mode, values in columns.items():
                checkpoints[name][category][f"{mode}_both_clocks_all_slots"] = int(sum(values))
                profiles[run][category, mode] = values
    recipe = json.loads((study / "llama3-selection.json").read_text())["selected_recipe"]
    outcomes = {}
    for category in ("flip", "agree", "unavailable"):
        families = [r["project"] for r in cases if r["category"] == category]
        outcomes[category] = {"cases": len(families), "projects": len(set(families))}
        for mode in MODES:
            repeats = {arm: np.stack([profiles[seed, recipe, arm][category, mode] for seed in SEEDS]) for arm in ARMS}
            outcomes[category][mode] = {
                "mean_accuracy": {arm: float(rows.mean()) for arm, rows in repeats.items()},
                "untouched": float(np.mean(profiles[0, "untouched", "pretrained"][category, mode])),
                "kam_minus_dot": family_difference(repeats["kam"], repeats["dot"], families),
            }
    return {"manifest_sha256": digest(manifest), "outcome_sha256": hashes,
            "checkpoints": checkpoints, "accuracy": outcomes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "manifest", "panel", "study", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write_json(args.output, summarize(args.work, args.manifest, args.panel, args.study))
