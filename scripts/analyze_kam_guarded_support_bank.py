#!/usr/bin/env python3
"""Audit the persistent support bank, its matched comparator and frozen hard reads."""

import argparse
import json
from pathlib import Path
from statistics import median

from analyze_kam_clock_archive_controls import estimate
from analyze_kam_guarded_fp32_sdpa import load
from analyze_kam_guarded_sdpa_training import paired
from analyze_kam_temporal_projection_control import clean_cases


def audit(rows, references):
    prior = {x["seed"]: x for x in references}
    assert len(rows) == len(prior) == 3 and sorted(x["seed"] for x in rows) == sorted(prior)
    assert all(x["steps"] == 3200 and x["device"] == "NVIDIA L4" for x in rows)
    assert all(x["parameters"] == rows[0]["parameters"] and x["archives_sha256"] == rows[0]["archives_sha256"] for x in rows)
    result = {}
    for geometry in ("dot", "radial"):
        bank = f"{geometry}/support_bank"
        point = f"{geometry}/pointwise"
        base = f"{geometry}/context"
        variants = {"context": base, "supports": bank, "pointwise": point}
        interventions = {"zero_support": bank, "top_one": bank, "top_four": bank}
        fixed = dict(zip(interventions, ("0", "1", "4"), strict=True))
        view = []
        for row in rows:
            source = prior[row["seed"]]["arms"][f"{geometry}/hard_clock_and_rank"]
            assert row["arms"][base]["fit"]["losses"] == source["fit"]["losses"]
            assert all(row["arms"][base]["soft"][name] == source[name] for name in ("generated", "archives"))
            outputs = {k: row["arms"][v]["soft"] for k, v in variants.items()}
            outputs.update({k: row["arms"][v]["frozen_interventions"][fixed[k]] for k, v in interventions.items()})
            view.append({"seed": row["seed"], "arms": outputs})
        panels = {}
        for label in (*variants, *interventions):
            opponent = "supports" if label in interventions else "context"
            archived = {}
            for topic in rows[0]["archives_sha256"]:
                for depth in (16, 48):
                    found = clean_cases(view, label, topic, depth)
                    control = clean_cases(view, opponent, topic, depth)
                    same = estimate(control, found, "joint", months=topic != "dallas")
                    archived[f"{topic}/{depth}"] = {"strict_exact": sum(r["joint"] for r in found.values()), "tested": len(found),
                                                     "relative_to": opponent, "interval": same}
                    if label == "supports":
                        mlp = clean_cases(view, "pointwise", topic, depth)
                        archived[f"{topic}/{depth}"]["vs_pointwise"] = estimate(mlp, found, "joint", months=topic != "dallas")
            synthetic = {}
            for kind, lengths in view[0]["arms"][label]["generated"].items():
                for d in lengths:
                    records = []
                    for row in view:
                        b, a = (row["arms"][name]["generated"][kind][d] for name in (opponent, label))
                        assert b["fixture_sha256"] == a["fixture_sha256"]
                        records.extend(zip(map(all, b["outcomes"]), map(all, a["outcomes"]), strict=True))
                    synthetic[f"{kind}/{d}"] = paired(*map(list, zip(*records, strict=True))) | {"relative_to": opponent}
            timing = median(row["arms"][variants[label]]["fit"]["seconds"] for row in rows) if label in variants else None
            panels[label] = {"archived": archived, "synthetic": synthetic, "training_median_s": timing,
                             "parameters": rows[0]["parameters"][(variants | interventions)[label]]}
        result[geometry] = panels
    return {"matched_seed_count": len(rows), "input_tokens_per_fit": 10556800, "models": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, nargs=3, required=True)
    p.add_argument("--references", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(list(map(load, args.measurements)), list(map(load, args.references)))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
