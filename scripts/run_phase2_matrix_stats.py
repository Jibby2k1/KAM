from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kam.stats import holm_adjust, paired_bootstrap_ci, paired_permutation_pvalue


COMPARISONS = [
    ("D0", "R0", "radial_context"),
    ("D0", "DD-v", "persistent_values"),
    ("D0", "DD-a", "persistent_routes"),
    ("D0", "DD-b", "persistent_both"),
    ("DD-v", "DR-v", "radial_memory_values"),
    ("DD-a", "DR-a", "radial_memory_routes"),
    ("DD-b", "DR-b", "radial_memory_both"),
    ("DD-b", "RR-b", "radial_context_with_memory"),
]


def load_rows(root: Path) -> list[dict[str, object]]:
    rows = [dict(row) for row in csv.DictReader((root / "all_metrics.csv").open(encoding="utf-8"))]
    for row in rows:
        payload = json.loads((root / str(row["run_id"]) / "metrics.json").read_text(encoding="utf-8"))
        row["mse"] = float(payload["final_validation"]["mse"])
        row["parameter_count"] = int(payload["parameter_count"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute suffix-aware paired statistics for the Phase II dynamic matrix.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.root)
    index = {(str(row["task"]), int(row["seed"]), str(row["variant"])): row for row in rows}
    result: list[dict[str, object]] = []
    for task in sorted({str(row["task"]) for row in rows}):
        seeds = sorted({int(row["seed"]) for row in rows if str(row["task"]) == task})
        for baseline, candidate, claim in COMPARISONS:
            paired = [
                (index[(task, seed, baseline)], index[(task, seed, candidate)])
                for seed in seeds
                if (task, seed, baseline) in index and (task, seed, candidate) in index
            ]
            if not paired:
                continue
            base_values = [float(base["mse"]) for base, _candidate in paired]
            candidate_values = [float(candidate["mse"]) for _base, candidate in paired]
            low, high = paired_bootstrap_ci(base_values, candidate_values, seed=7)
            result.append({
                "task": task,
                "claim": claim,
                "baseline": baseline,
                "candidate": candidate,
                "pairs": len(paired),
                "parameter_count": sorted({int(base["parameter_count"]) for base, _candidate in paired}),
                "baseline_mean_mse": sum(base_values) / len(base_values),
                "candidate_mean_mse": sum(candidate_values) / len(candidate_values),
                "mean_improvement": sum(base - cand for base, cand in zip(base_values, candidate_values)) / len(paired),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "permutation_p": paired_permutation_pvalue(base_values, candidate_values, seed=7),
            })
    adjusted = holm_adjust([float(row["permutation_p"]) for row in result])
    for row, value in zip(result, adjusted):
        row["holm_p"] = float(value)
        row["ci_excludes_zero"] = bool(float(row["bootstrap_ci_low"]) > 0 or float(row["bootstrap_ci_high"]) < 0)
        row["parameter_count"] = row["parameter_count"][0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in result for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {len(result)} suffix-aware paired comparisons to {args.output}")


if __name__ == "__main__":
    main()
