from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from kam.stats import holm_adjust, paired_bootstrap_ci, paired_permutation_pvalue


COMPARISONS = [
    ("D0", "DD-b", "persistent_memory"),
    ("DD-b", "DR-b", "radial_memory"),
    ("D0", "R0", "radial_context"),
]


def finite(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if number == number and abs(number) != float("inf") else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate held-out schedule transitions within seed and compute paired NLMS effects.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = list(csv.DictReader(args.metrics.open(encoding="utf-8")))
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[(str(row["task"]), int(row["seed"]), str(row["variant"]))].append(row)
    aggregates: dict[tuple[str, int, str], dict[str, float | str | int]] = {}
    for key, rows in grouped.items():
        fields: dict[str, float] = {}
        for field in ("early_loss", "late_loss", "recovery_steps", "forgetting_ratio"):
            values = [value for row in rows if (value := finite(row.get(field, ""))) is not None]
            fields[field] = sum(values) / len(values) if values else float("nan")
        task, seed, variant = key
        aggregates[key] = {"task": task, "seed": seed, "variant": variant, "n_transitions": len(rows), **fields}
    results: list[dict[str, object]] = []
    for task in sorted({str(row["task"]) for row in raw}):
        seeds = sorted({int(row["seed"]) for row in raw if str(row["task"]) == task})
        for baseline, candidate, claim in COMPARISONS:
            paired = [(aggregates[(task, seed, baseline)], aggregates[(task, seed, candidate)]) for seed in seeds if (task, seed, baseline) in aggregates and (task, seed, candidate) in aggregates]
            if not paired:
                continue
            base = [float(row["late_loss"]) for row, _candidate in paired]
            cand = [float(row["late_loss"]) for _row, row in paired]
            low, high = paired_bootstrap_ci(base, cand, seed=7)
            results.append({
                "task": task,
                "adapter": "nlms",
                "claim": claim,
                "baseline": baseline,
                "candidate": candidate,
                "pairs": len(paired),
                "mean_baseline_late_loss": sum(base) / len(base),
                "mean_candidate_late_loss": sum(cand) / len(cand),
                "mean_improvement": sum(left - right for left, right in zip(base, cand)) / len(base),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "permutation_p": paired_permutation_pvalue(base, cand, seed=7),
            })
    adjusted = holm_adjust([float(row["permutation_p"]) for row in results])
    for row, value in zip(results, adjusted):
        row["holm_p"] = float(value)
        row["ci_excludes_zero"] = bool(float(row["bootstrap_ci_low"]) > 0 or float(row["bootstrap_ci_high"]) < 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in results for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    args.output.with_suffix(".json").write_text(json.dumps({"aggregates": list(aggregates.values()), "comparisons": results}, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} held-out paired comparisons from {len(raw)} transition rows to {args.output}")


if __name__ == "__main__":
    main()
