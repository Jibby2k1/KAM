from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from kam.stats import holm_adjust, paired_bootstrap_ci, paired_permutation_pvalue


def _metric_rows(path: Path) -> list[dict[str, object]]:
    rows = [dict(row) for row in csv.DictReader(path.open(encoding="utf-8"))]
    for row in rows:
        metrics_path = path.parent / str(row["run_id"]) / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row["mse"] = payload.get("final_validation", {}).get("mse")
        row["mae"] = payload.get("final_validation", {}).get("mae")
    return rows


def _float(value: object) -> float:
    return float(value)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired Phase II effects with bootstrap/permutation inference.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/phase2/paired_screen/paired_stats.csv"))
    args = parser.parse_args()
    rows = _metric_rows(args.metrics)
    index: dict[tuple[str, int, str], dict[str, object]] = {}
    for row in rows:
        index[(str(row["task"]), int(row["seed"]), str(row["variant"]))] = row
    comparisons = [("R0", "D0"), ("RR", "DD"), ("DR", "DD")]
    result_rows: list[dict[str, object]] = []
    for task in sorted({str(row["task"]) for row in rows}):
        seeds = sorted({int(row["seed"]) for row in rows if row["task"] == task})
        for variant, baseline in comparisons:
            paired = [(seed, index[(task, seed, baseline)], index[(task, seed, variant)]) for seed in seeds if (task, seed, baseline) in index and (task, seed, variant) in index]
            if not paired:
                continue
            baseline_mse = [_float(base["mse"]) for _seed, base, _candidate in paired]
            variant_mse = [_float(candidate["mse"]) for _seed, _base, candidate in paired]
            ci_low, ci_high = paired_bootstrap_ci(baseline_mse, variant_mse, seed=7)
            result_rows.append({
                "task": task,
                "metric": "mse",
                "baseline": baseline,
                "variant": variant,
                "pairs": len(paired),
                "baseline_mean": sum(baseline_mse) / len(baseline_mse),
                "variant_mean": sum(variant_mse) / len(variant_mse),
                "mean_improvement": sum(b - v for b, v in zip(baseline_mse, variant_mse)) / len(paired),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "permutation_p": paired_permutation_pvalue(baseline_mse, variant_mse, seed=7),
            })
    adjusted = holm_adjust([float(row["permutation_p"]) for row in result_rows]) if result_rows else []
    for row, value in zip(result_rows, adjusted):
        row["holm_p"] = float(value)
        row["ci_excludes_zero"] = bool(float(row["bootstrap_ci_low"]) > 0 or float(row["bootstrap_ci_high"]) < 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in result_rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)
    args.output.with_suffix(".json").write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(result_rows)} paired comparisons to {args.output}")


if __name__ == "__main__":
    main()
