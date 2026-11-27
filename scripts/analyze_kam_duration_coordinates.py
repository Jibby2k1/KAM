#!/usr/bin/env python3
"""Audit coordinate factorization and token-matched length exposure."""

import argparse
import json
from pathlib import Path
from statistics import mean

from analyze_kam_duration_pointer import CONDITIONS, compare_archive
from analyze_kam_shared_serving import unpack

SEEDS = (542981, 543001, 543037)
MODELS = ("baseline", "scaled_features", "scaled_positions", "scaled_both", "wide_baseline", "wide_scaled_both")
METHODS = (*MODELS, "mean_duration", "exact_prefix")
SHAPES = (*CONDITIONS, "256/90")
SHORT = [[16, 64], [32, 32]]
WIDE = [[16, 70], [32, 35], [64, 16], [128, 7]]


def inspect(raw, fixture):
    assert raw["device"] == "NVIDIA L4" and raw["steps"] == 2400 and raw["bank"] == 16_384
    assert raw["fixture_sha256"] == fixture["fixture_sha256"]
    training = raw["training"]
    assert set(training) == set(MODELS)
    short_hashes = training["baseline"]["bank_sha256"]
    wide_hashes = training["wide_baseline"]["bank_sha256"]
    assert len(short_hashes) == 2 and len(wide_hashes) == 4
    assert set(short_hashes).isdisjoint(wide_hashes)
    for mode, result in training.items():
        wide = mode.startswith("wide_")
        assert result["schedule"] == (WIDE if wide else SHORT)
        assert result["bank_sha256"] == (wide_hashes if wide else short_hashes)
        assert result["tokens"] == 2_572_800 and result["seconds"] > 0
        assert [p["step"] for p in result["losses"]] == [100, 400, 800, 1600, 2400]
        assert all(0 <= p["loss"] < float("inf") for p in result["losses"])
    synthetic = {}
    assert set(raw["synthetic"]) == set(SHAPES)
    for shape in SHAPES:
        panel = raw["synthetic"][shape]
        assert len(panel["gold"]) == 512 and set(panel["predictions"]) == set(METHODS)
        assert panel["predictions"]["exact_prefix"] == panel["gold"]
        depth = int(shape.split("/")[0])
        assert all(0 <= pred < depth for predictions in panel["predictions"].values() for pred in predictions)
        synthetic[shape] = {m: sum(x == y for x, y in zip(panel["predictions"][m], panel["gold"], strict=True))
                            for m in METHODS}
    cases, predictions = fixture["rows"], raw["archived"]
    assert len(cases) == 1173 and [r["id"] for r in cases] == [r["id"] for r in predictions]
    archived = compare_archive(cases, predictions, modes=METHODS)
    return {"train": training, "synthetic_correct": synthetic, "archive_correct": archived}


def summarize(runs, fixture):
    records = {r["seed"]: r for r in runs}
    assert len(runs) == len(records) == len(SEEDS) and set(records) == set(SEEDS)
    assert all(len({r["synthetic"][shape]["sha256"] for r in runs}) == len(SEEDS) for shape in SHAPES)
    panels = {str(seed): inspect(records[seed], fixture) for seed in SEEDS}
    synthetic = {s: {m: round(mean(v["synthetic_correct"][s][m] / 512 for v in panels.values()), 8)
                     for m in METHODS} for s in SHAPES}
    metrics = panels[str(SEEDS[0])]["archive_correct"]["baseline"]
    archive = {m: {key: round(mean(v["archive_correct"][m][key] / 1173 for v in panels.values()), 8)
                   for key in metrics} for m in METHODS}
    return {"seeds": panels, "synthetic_mean_accuracy": synthetic, "archived_mean_rates": archive}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs=3, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = summarize([unpack(f) for f in args.runs], unpack(args.manifest))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
