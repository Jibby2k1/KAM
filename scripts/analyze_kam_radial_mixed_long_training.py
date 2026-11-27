#!/usr/bin/env python3
"""Audit equal-token long radial training against the matched short curriculum."""

import argparse
import json
from pathlib import Path
from statistics import mean

from analyze_kam_radial_archive_training import EVALS, MODES, SEEDS, TOPICS, fingerprint_panels, summarize
from analyze_kam_shared_serving import unpack

SCHEDULE = [[16, 125], [48, 47], [128, 16], [512, 4]]
GENERATED = ("common_fp32_generated", "native_generated")
DEPTHS = ("20", "128", "512")


def synthetic_summary(runs):
    anchors, outcomes = {}, {}
    for hardware, jobs in runs.items():
        metrics = {}
        for run in jobs:
            seed = run["seed"]
            assert run["bank_shape_counts"] == [16_384, 16_384, 8192, 2048]
            assert len(run["bank_sha256"]) == len(run["index_sha256"]) == 4
            eager = run["modes"]["eager_fp32"]
            assert eager["common_fp32_generated"] == eager["native_generated"]
            stamps = {d: value["fixture_sha256"] for d, value in eager["common_fp32_generated"].items()}
            assert set(stamps) == set(DEPTHS) and anchors.setdefault(seed, stamps) == stamps
            scores = {}
            for mode in MODES:
                condition = run["modes"][mode]
                scores[mode] = {}
                for evaluate in GENERATED:
                    assert set(condition[evaluate]) == set(DEPTHS)
                    panel = condition[evaluate]
                    assert all(panel[d]["fixture_sha256"] == stamps[d] for d in DEPTHS)
                    assert all(len(v["correct"]) == 512 and all(isinstance(c, bool) for c in v["correct"])
                               for v in panel.values())
                    scores[mode][evaluate] = {d: sum(p["correct"]) for d, p in panel.items()}
            metrics[str(seed)] = scores
        means = {mode: {view: {d: round(mean(metrics[str(s)][mode][view][d] / 512 for s in SEEDS), 8)
                               for d in DEPTHS} for view in GENERATED} for mode in MODES}
        outcomes[hardware] = {"correct_per_seed": metrics, "mean_joint": means}
    return outcomes


def check_short_match(short, wide):
    for device in ("b200", "l4"):
        earlier = {run["seed"]: run for run in short[device]}
        assert set(earlier) == set(SEEDS)
        for run in wide[device]:
            other = earlier[run["seed"]]
            assert all(run[k] == other[k] for k in ("order", "device", "start_sha256", "source_sha256"))
            assert run["bank_sha256"][:2] == other["bank_sha256"]
            assert run["index_sha256"][:2] != other["index_sha256"]
            for cohort in TOPICS:
                a = run["modes"]["eager_fp32"]["common_fp32_eval"][cohort]
                b = other["modes"]["eager_fp32"]["common_fp32_eval"][cohort]
                assert fingerprint_panels(a, cohort) == fingerprint_panels(b, cohort)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b200", required=True, type=Path, nargs=3)
    p.add_argument("--l4", required=True, type=Path, nargs=3)
    p.add_argument("--short-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    wide = {k: [unpack(f) for f in getattr(a, k)] for k in ("b200", "l4")}
    short = {k: [unpack(f) for f in sorted(a.short_root.glob(f"train-{k}-*.json.gz"))] for k in ("b200", "l4")}
    check_short_match(short, wide)
    archived = summarize(wide, schedule=SCHEDULE)
    earlier = summarize(short)
    changes = {}
    for hardware in archived:
        changes[hardware] = {}
        for mode in MODES:
            start, end = (runs[hardware]["means"][mode] for runs in (earlier, archived))
            changes[hardware][mode] = {p: {k: {d: round(end[p][k][d]["joint"] - start[p][k][d]["joint"], 8)
                                              for d in ("16", "48")} for k in TOPICS} for p in EVALS}
    result = {"archived": archived, "generated": synthetic_summary(wide), "archive_joint_change_from_short": changes}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
