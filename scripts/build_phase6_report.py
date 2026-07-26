#!/usr/bin/env python3
"""Build a compact, descriptive Phase 6 report from row-level JSON outputs.

The report intentionally summarizes measured groups without treating a Latin-
hypercube screen as a replicated statistical comparison. Inferential claims
belong in the locked confirmation analysis after seed pairing and held-out
evaluation are complete.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kam.phase6.plots import plot_learning_curves


STAGE_SETTINGS: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]] = {
    "stage1_mechanism": (
        ("task", "architecture"),
        ("task", "optimizer"),
        ("architecture", "geometry"),
        "initial_loss loss training_steps alternating_geometry_steps solver_condition_number measured_forward_ms peak_vram_bytes",
    ),
    "stage2_transformer_comparison": (
        ("architecture", "scale"),
        ("task", "architecture"),
        ("architecture", "task"),
        "initial_loss loss perplexity training_steps training_tokens total_parameters trainable_parameters active_parameter_count target_parameter_budget parameter_match_error_fraction declared_memory_slots effective_memory_slots effective_top_k geometry_update_steps algebra_update_steps measured_forward_ms decoder_throughput_tokens_per_sec peak_vram_bytes",
    ),
    "stage3_router_scaling": (
        ("router", "precision"),
        ("router", "benchmark_supports"),
        ("router", "top_k"),
        "recall_at_k_against_exact routing_forward_ms routing_throughput_tokens_per_sec peak_vram_bytes bank_storage_bytes optimizer_state_bytes effective_support_count dead_support_fraction load_balance_error",
    ),
    "stage4_online_adaptation": (
        ("architecture", "adapter"),
        ("stream_task", "adapter"),
        ("architecture", "stream_task"),
        "global_nmse early_nmse late_nmse reacquisition_time geometry_update_count memory_used episodic_active",
    ),
    "stage5_long_training": (
        ("architecture", "scale"),
        ("task", "architecture"),
        ("architecture", "task"),
        "initial_loss loss perplexity training_steps training_tokens declared_token_budget budget_completion_fraction total_parameters active_parameter_count target_parameter_budget parameter_match_error_fraction declared_memory_slots effective_memory_slots effective_top_k measured_forward_ms decoder_throughput_tokens_per_sec",
    ),
    "stage6_confirmation": (
        ("claim", "architecture"),
        ("task", "architecture"),
        ("architecture", "task"),
        "initial_loss loss perplexity training_steps training_tokens total_parameters active_parameter_count target_parameter_budget parameter_match_error_fraction declared_memory_slots effective_memory_slots effective_top_k geometry_update_steps algebra_update_steps measured_forward_ms decoder_throughput_tokens_per_sec peak_vram_bytes",
    ),
}


def _load_rows(run_root: Path) -> list[dict[str, Any]]:
    combined = run_root / "all_metrics.jsonl"
    paths = [combined] if combined.exists() else sorted(run_root.glob("row_*.json")) + sorted(run_root.glob("row_*.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric(row: dict[str, Any], field: str) -> Any:
    if field in row.get("metrics", {}):
        return row["metrics"][field]
    if f"metric_{field}" in row:
        return row[f"metric_{field}"]
    return row.get(field)


def _label(row: dict[str, Any], field: str) -> str:
    value = _metric(row, field)
    if value is None:
        return "unknown"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    magnitude = abs(value)
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e5):
        return f"{value:.3g}"
    return f"{value:.5g}"


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _table(rows: list[dict[str, Any]], dimensions: tuple[str, ...], metrics: list[str], *, limit: int = 80) -> list[str]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(_label(row, field) for field in dimensions)].append(row)
    ordered = sorted(groups.items(), key=lambda item: item[0])
    shown = ordered[:limit]
    headers = [*dimensions, "n", *metrics]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for labels, group in shown:
        values = [str(len(group))]
        for metric in metrics:
            values.append(_fmt(_mean(_metric(row, metric) for row in group)))
        lines.append("| " + " | ".join([*labels, *values]) + " |")
    if len(ordered) > limit:
        lines.append(f"| … | … | {len(ordered) - limit} additional groups omitted; use the machine-readable artifacts. |")
    return lines


def _metric_names(stage: str, rows: list[dict[str, Any]], declared: str) -> list[str]:
    available = {key for row in rows for key in row.get("metrics", {})}
    available.update(key.removeprefix("metric_") for row in rows for key in row if key.startswith("metric_"))
    return [field for field in declared.split() if field in available]


def build_report(stage: str, run_root: Path, output: Path, *, manifest: Path | None = None) -> dict[str, Any]:
    rows = _load_rows(run_root)
    settings = STAGE_SETTINGS.get(stage)
    if settings is None:
        raise ValueError(f"unsupported Phase 6 stage: {stage}")
    dimensions_a, dimensions_b, dimensions_c, declared_metrics = settings
    passed = [row for row in rows if row.get("status") == "pass"]
    failures = [row for row in rows if row.get("status") != "pass"]
    metric_names = _metric_names(stage, passed, declared_metrics)
    counts = {
        field: dict(sorted(Counter(_label(row, field) for row in rows).items()))
        for field in ("task", "architecture", "optimizer", "geometry", "router", "precision", "adapter", "scale", "claim")
        if any(_metric(row, field) is not None for row in rows)
    }
    artifact_manifest = run_root / "artifact_manifest.json"
    if stage == "stage4_online_adaptation":
        histories = [
            [float(value) for value in _metric(row, "squared_error_history")]
            for row in passed
            if isinstance(_metric(row, "squared_error_history"), list)
            and _metric(row, "squared_error_history")
            and all(_finite(value) is not None for value in _metric(row, "squared_error_history"))
        ]
        if histories:
            width = max(len(history) for history in histories)
            mean_curve = [
                sum(history[index] for history in histories if index < len(history))
                / sum(index < len(history) for history in histories)
                for index in range(width)
            ]
            plot_learning_curves({"mean_squared_error": mean_curve}, run_root / "adaptation_curves.png")
    plot_names = sorted(path.name for path in run_root.glob("*.png"))
    gate = "PASS" if rows and not failures else "BLOCKED"
    lines = [
        f"# Phase 6 {stage.replace('_', ' ').title()} report",
        "",
        f"- Row outputs: **{len(rows)}**",
        f"- Passing rows: **{len(passed)}**",
        f"- Failed/non-passing rows: **{len(failures)}**",
        f"- Row-level execution gate: **{gate}**",
        f"- Run root: `{run_root}`",
        f"- Manifest: `{manifest}`" if manifest else "- Manifest: not supplied",
        f"- Artifact manifest: `{artifact_manifest}`" if artifact_manifest.exists() else "- Artifact manifest: not present",
        "",
        "This is a descriptive screen. It does not establish a paired treatment effect, a scaling law, or a promotion decision; those require the declared seed-paired and held-out analyses.",
        "",
        "## Factor coverage",
        "",
    ]
    for field, values in counts.items():
        lines.append(f"- `{field}`: " + ", ".join(f"{key}={value}" for key, value in values.items()))
    for title, dimensions in (("Primary grouped summary", dimensions_a), ("Task/group summary", dimensions_b), ("Cross-check summary", dimensions_c)):
        lines.extend(["", f"## {title}", ""])
        lines.extend(_table(passed, dimensions, metric_names))
    if failures:
        lines.extend(["", "## Failures", "", "The following row IDs failed; inspect their JSON error fields before retrying:", ""])
        lines.extend(f"- `{row.get('row_id', 'unknown')}`: {row.get('error', 'unknown error')}" for row in failures[:50])
    if plot_names:
        lines.extend(["", "## Generated figures", ""])
        lines.extend(f"- `{name}`" for name in plot_names)
    lines.extend(["", "## Interpretation guardrail", "", "Use this report to locate promising factor/resource combinations and failure modes. Do not promote a configuration from this screen alone; preserve the locked claims, inferential unit, equivalence margins, and held-out data specified by the Phase 6 brief.", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return {"stage": stage, "rows": len(rows), "passed": len(passed), "failed": len(failures), "output": str(output), "metrics": metric_names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(build_report(args.stage, args.run_root, args.output, manifest=args.manifest), indent=2))


if __name__ == "__main__":
    main()
