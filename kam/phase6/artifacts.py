from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_ARTIFACTS = (
    "run_manifest",
    "all_metrics",
    "paired_seed_metrics",
    "router_metrics",
    "geometry_drift",
    "algebra_solver_metrics",
    "support_diagnostics",
    "adaptation_metrics",
    "scaling_metrics",
    "confirmatory_metrics",
    "pareto_frontier",
)


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in row.items() if key != "metrics"}
    result.update({f"metric_{key}": value for key, value in row.get("metrics", {}).items()})
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objectives = [field for field in ("metric_loss", "metric_measured_forward_ms", "metric_peak_vram", "metric_active_parameter_count") if any(field in row for row in rows)]
    if not objectives:
        return rows
    numeric = [(row, [float(row.get(field, float("inf"))) for field in objectives]) for row in rows]
    frontier: list[dict[str, Any]] = []
    for candidate, values in numeric:
        dominated = any(all(other_values[index] <= values[index] for index in range(len(objectives))) and any(other_values[index] < values[index] for index in range(len(objectives))) for other, other_values in numeric if other is not candidate)
        if not dominated:
            frontier.append(candidate)
    return frontier


def write_artifacts(rows: list[dict[str, Any]], root: str | Path) -> dict[str, Any]:
    """Write canonical JSONL artifacts and optional true Parquet exports.

    JSONL is retained as the portable canonical format when the environment
    lacks a Parquet engine; the manifest records that fact instead of writing
    a mislabeled non-Parquet file.
    """
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    flat = [_flatten(row) for row in rows]
    records: dict[str, Any] = {"jsonl": {}, "parquet": {}, "parquet_engine": None}
    for artifact in REQUIRED_ARTIFACTS:
        path = destination / f"{artifact}.jsonl"
        selected = flat
        if artifact == "pareto_frontier":
            selected = _pareto_frontier(flat)
        if artifact == "router_metrics":
            selected = [row for row in flat if any(token in str(row.get("stage", "")) for token in ("router", "stage3")) or "metric_routing_entropy" in row]
        elif artifact == "adaptation_metrics":
            selected = [row for row in flat if "adapt" in str(row.get("stage", "")) or "adapter" in row]
        elif artifact == "support_diagnostics":
            selected = [row for row in flat if "metric_effective_support_count" in row or "metric_dead_support_fraction" in row]
        _write_jsonl(path, selected)
        records["jsonl"][artifact] = str(path)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        records["parquet_engine"] = "pyarrow"
        for artifact in REQUIRED_ARTIFACTS:
            source = [json.loads(line) for line in (destination / f"{artifact}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            parquet_path = destination / f"{artifact}.parquet"
            pq.write_table(pa.Table.from_pylist(source), parquet_path)
            records["parquet"][artifact] = str(parquet_path)
    except Exception as exc:  # noqa: BLE001 - optional dependency is environment-specific
        records["parquet_engine"] = f"unavailable: {type(exc).__name__}"
    (destination / "artifact_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


__all__ = ["REQUIRED_ARTIFACTS", "write_artifacts"]
