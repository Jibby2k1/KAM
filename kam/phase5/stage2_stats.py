"""Paired Stage 2 effects with training seed as inferential unit."""
from __future__ import annotations
import csv
import math
import statistics
from collections import defaultdict
from typing import Any


def paired_effects(rows: list[dict[str, Any]], baseline: str = "D0") -> list[dict[str, Any]]:
    index = {(str(row.get("task")), str(row.get("cell")), str(row.get("seed_index")), str(row.get("variant"))): row for row in rows}
    variants = sorted({str(row.get("variant")) for row in rows if str(row.get("variant")) != baseline})
    output = []
    for variant in variants:
        for key in sorted({(str(row.get("task")), str(row.get("cell")), str(row.get("seed_index"))) for row in rows}):
            control = index.get((*key, baseline))
            treatment = index.get((*key, variant))
            if not control or not treatment:
                continue
            base = float(control.get("heldout_nmse", control.get("test_nmse", "nan")))
            value = float(treatment.get("heldout_nmse", treatment.get("test_nmse", "nan")))
            if not math.isfinite(base) or not math.isfinite(value):
                continue
            output.append({"task": key[0], "cell": key[1], "seed_index": key[2], "baseline": baseline, "variant": variant, "baseline_nmse": base, "variant_nmse": value, "relative_improvement": (base - value) / max(abs(base), 1e-12)})
    return output


def write_effects(path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["task", "cell", "seed_index", "baseline", "variant", "baseline_nmse", "variant_nmse", "relative_improvement"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
