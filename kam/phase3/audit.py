from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from .table import write_json, write_table


RELEVANT_SUFFIXES = {".pt", ".json", ".csv", ".npz", ".sqlite", ".yaml", ".yml", ".py", ".md", ".png"}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _category(path: Path) -> str:
    name = path.name
    if name == "best_model.pt":
        return "checkpoint"
    if name == "resolved_config.json":
        return "resolved_config"
    if name == "metrics.json":
        return "metrics"
    if "transition" in name or "shift" in name:
        return "transition_trace"
    if "deletion" in name:
        return "deletion"
    if "search" in name or name.endswith(".sqlite"):
        return "search_database"
    if path.suffix in {".py", ".yaml", ".yml"}:
        return "source_or_config"
    return "artifact"


def _checkpoint_row(path: Path, root: Path) -> dict[str, Any]:
    run_dir = path.parent
    metrics_path = run_dir / "metrics.json"
    resolved_path = run_dir / "resolved_config.json"
    metrics: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
    if resolved_path.exists():
        try:
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            resolved = {}
    run = resolved.get("run", {}) if isinstance(resolved.get("run"), dict) else {}
    model_spec = metrics.get("model_spec", resolved.get("model_spec", {}))
    if not isinstance(model_spec, dict):
        model_spec = {}
    return {
        "path": str(path.relative_to(root)),
        "run_id": metrics.get("run_id", run.get("run_id", run_dir.name)),
        "task": metrics.get("task", run.get("task")),
        "variant": metrics.get("variant", run.get("variant", model_spec.get("model_name"))),
        "seed": metrics.get("seed", run.get("seed")),
        "parameter_count": metrics.get("parameter_count", model_spec.get("parameter_match_target")),
        "training_steps": metrics.get("training_steps", run.get("steps")),
        "status": metrics.get("status", "unknown"),
        "git_commit": resolved.get("git_commit"),
        "git_dirty": resolved.get("git_dirty"),
        "has_metrics": metrics_path.exists(),
        "has_resolved_config": resolved_path.exists(),
    }


def _headline_consistency(root: Path, checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    phase2 = root / "results" / "phase2"

    def count_metrics(name: str) -> int:
        directory = phase2 / name
        return sum(1 for path in directory.glob("*/metrics.json")) if directory.exists() else 0

    def count_csv_rows(path: Path) -> int:
        if not path.exists():
            return 0
        return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)

    expected = {
        "stationary_runs": 50,
        "switching_runs": 50,
        "exact_capacity_runs": 90,
        "heldout_transition_rows": 1260,
    }
    observed = {
        "checkpoint_count": len(checkpoints),
        "phase2_report_present": (root / "reports" / "phase2" / "PHASE2_RESULTS.md").exists(),
        "stationary_runs": count_metrics("paired_screen"),
        "switching_runs": count_metrics("switching_paired_screen"),
        "exact_capacity_runs": count_metrics("dynamic_matrix_pmatched"),
        "heldout_transition_rows": count_csv_rows(phase2 / "heldout_dynamic_matrix_nlms" / "shift_metrics.csv"),
    }
    for key, number in expected.items():
        observed[f"{key}_expected"] = number
        observed[f"{key}_match"] = observed[key] == number
    observed["headline_matches"] = all(observed[f"{key}_match"] for key in expected)
    return observed


def run_audit(root: str | Path = ".", output_root: str | Path = "results/phase3") -> dict[str, Any]:
    root = Path(root).resolve()
    output_root = Path(output_root)
    report_root = root / "reports" / "phase3"
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    search_roots = [root / "results" / "phase2", root / "outputs", root / "configs" / "phase2", root / "kam", root / "scripts"]
    inventory: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob("*")):
            if not path.is_file() or path.suffix not in RELEVANT_SUFFIXES:
                continue
            try:
                relative = str(path.relative_to(root))
                inventory.append({
                    "path": relative,
                    "category": _category(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "modified_time": path.stat().st_mtime,
                })
                if path.name == "best_model.pt":
                    checkpoints.append(_checkpoint_row(path, root))
            except (OSError, ValueError):
                continue
    summary = {
        "generated_at": time.time(),
        "workspace": str(root),
        "inventory_rows": len(inventory),
        "checkpoint_rows": len(checkpoints),
        "complete_checkpoints": sum(row.get("status") == "complete" for row in checkpoints),
        "tasks": sorted({str(row.get("task")) for row in checkpoints if row.get("task")}),
        "variants": sorted({str(row.get("variant")) for row in checkpoints if row.get("variant")}),
        "headline_consistency": _headline_consistency(root, checkpoints),
    }
    write_table(output_root / "input_inventory.parquet", inventory)
    write_table(output_root / "checkpoint_inventory.parquet", checkpoints)
    write_json(output_root / "audit_summary.json", summary)
    lines = [
        "# Phase II Evidence Audit",
        "",
        f"Generated from `{root}`. The audit hashed {len(inventory):,} relevant files and found {len(checkpoints):,} checkpoint artifacts.",
        "",
        "## Inventory",
        "",
        f"- Complete checkpoints: **{summary['complete_checkpoints']:,}**",
        f"- Tasks represented: `{', '.join(summary['tasks']) or 'none'}`",
        f"- Variants represented: `{', '.join(summary['variants']) or 'none'}`",
        f"- Machine-readable inventory: `results/phase3/input_inventory.csv` and `results/phase3/checkpoint_inventory.csv` when Parquet support is unavailable.",
        "",
        "## Headline consistency",
        "",
    ]
    for key, value in summary["headline_consistency"].items():
        if key.endswith("_match") or key.endswith("_expected"):
            continue
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "The headline consistency values are a reconciliation flag, not a replacement for the raw-file reconstruction. Any false value must be resolved before confirmatory claims.",
        "",
    ])
    (report_root / "PHASE2_EVIDENCE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase II artifacts for Phase III reuse.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("results/phase3"))
    args = parser.parse_args()
    print(json.dumps(run_audit(args.root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
