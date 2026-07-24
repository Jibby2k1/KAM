from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .aggregate import collect_metrics
from .audit import run_audit
from .table import read_json, write_json


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _pair_variant(rows: list[dict[str, Any]], left_variant: str, right_variant: str, metric: str = "mse") -> list[tuple[float, float]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("task"), row.get("scale"), row.get("trial"), row.get("seed"))
        grouped.setdefault(key, {})[str(row.get("variant"))] = row
    pairs: list[tuple[float, float]] = []
    for variants in grouped.values():
        left = variants.get(left_variant)
        right = variants.get(right_variant)
        if left is None or right is None or not _finite(left.get(metric)) or not _finite(right.get(metric)):
            continue
        pairs.append((float(left[metric]), float(right[metric])))
    return pairs


def audit_gate(root: str | Path, output: str | Path) -> dict[str, Any]:
    summary = run_audit(root, Path(output).parent.parent if Path(output).parent.name == "gates" else "results/phase3")
    inventory_ok = summary.get("inventory_rows", 0) > 0
    checkpoints_ok = summary.get("complete_checkpoints", 0) > 0
    result = {
        "gate": "A",
        "pass": bool(inventory_ok and checkpoints_ok),
        "criteria": {
            "inventory_present": inventory_ok,
            "complete_checkpoints_present": checkpoints_ok,
            "headline_consistency": summary.get("headline_consistency", {}),
        },
        "summary": summary,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def primary_gate(aggregate_path: str | Path, output: str | Path, gate_config: str | Path | None = None) -> dict[str, Any]:
    aggregate = read_json(aggregate_path)
    effects = aggregate.get("effects", []) if isinstance(aggregate, dict) else []
    run_root = Path(aggregate.get("run_root", Path(aggregate_path).parent))
    rows, deletion_rows, stability_rows = collect_metrics(run_root)
    config: dict[str, Any] = {}
    if gate_config and Path(gate_config).exists():
        import yaml
        config = yaml.safe_load(Path(gate_config).read_text(encoding="utf-8")) or {}
    min_pairs = int(config.get("min_seed_pairs", 3))
    threshold = float(config.get("practical_improvement_threshold", 0.15))
    primary_rows = [row for row in effects if row.get("task") == "switching_mackey_glass" and row.get("scale") in {"XS", "S", "M", "L"}]
    primary_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in primary_rows:
        primary_groups[(row.get("task"), row.get("scale"), row.get("endpoint"))] = row
    primary_candidates = list(primary_groups.values())
    primary = max(primary_candidates, key=lambda row: int(row.get("group_n", 0)), default=None)
    primary_pass = bool(primary and int(primary.get("group_n", 0)) >= min_pairs and float(primary.get("group_relative_ci_low", -np.inf)) > 0 and float(primary.get("group_mean_relative_improvement", -np.inf)) >= threshold)

    random_pairs = _pair_variant(rows, "RF-b", "DD-b")
    random_improvements = np.asarray([left - right for left, right in random_pairs], dtype=float) if random_pairs else np.asarray([])
    random_pass = bool(random_improvements.size >= min_pairs and float(random_improvements.mean()) > 0)

    causal = [row for row in deletion_rows if row.get("variant") == "DD-b"]
    causal_top = np.asarray([float(row.get("top_delta", np.nan)) for row in causal], dtype=float)
    causal_random = np.asarray([float(row.get("random_delta", np.nan)) for row in causal], dtype=float)
    causal_pass = bool(causal_top.size and np.nanmean(causal_top - causal_random) > 0)

    dead_fractions = [float(row.get("dead_support_fraction")) for row in stability_rows if _finite(row.get("dead_support_fraction"))]
    stability_pass = bool(dead_fractions and max(dead_fractions) <= 0.25)
    result = {
        "gate": "primary",
        "pass": bool(primary_pass and random_pass and causal_pass and stability_pass),
        "criteria": {
            "primary_practical_and_ci": primary_pass,
            "learned_vs_random": random_pass,
            "top_supports_vs_random_deletion": causal_pass,
            "support_stability": stability_pass,
        },
        "primary": primary,
        "random_control": {"n": len(random_pairs), "mean_dd_minus_rf_loss_difference": float(random_improvements.mean()) if random_improvements.size else None},
        "causal": {"n": int(causal_top.size), "mean_top_minus_random_delta": float(np.nanmean(causal_top - causal_random)) if causal_top.size else None},
        "stability": {"n": len(dead_fractions), "max_dead_support_fraction": max(dead_fractions) if dead_fractions else None},
        "note": "A development gate result does not convert search rows into confirmatory evidence.",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    report_root = Path("reports/phase3")
    report_root.mkdir(parents=True, exist_ok=True)
    memo = [
        "# Phase III Decision Memo",
        "",
        f"Gate status: **{'PASS' if result['pass'] else 'HOLD/FAIL'}**.",
        "",
        "This memo preserves the preregistered distinction between development search and confirmation.",
        "",
        "1. Learned persistent memory versus no memory: evaluated by the paired D0/DD-b endpoint table.",
        f"2. Practical 15% threshold and positive CI: **{primary_pass}**.",
        f"3. Learned bank versus equal-dimensional random features: **{random_pass}**.",
        f"4. Top supports causally important: **{causal_pass}**.",
        f"5. Supports stable/noncollapsed: **{stability_pass}**.",
        "6. Radial memory comparison: inspect the DR-b rows; this gate does not promote radial memory automatically.",
        "7. Cost: inspect `all_metrics.parquet` fallback tables and the efficiency report.",
        f"8. Decision: **{'promote only to locked confirmation' if result['pass'] else 'narrow, gather more evidence, or stop rather than launch confirmation'}**.",
        "",
    ]
    (report_root / "PHASE3_DECISION_MEMO.md").write_text("\n".join(memo), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 3 scientific gate.")
    parser.add_argument("--aggregate", type=Path, default=None)
    parser.add_argument("--audit-summary", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--gate-type", choices=["audit", "primary"], default="primary")
    parser.add_argument("--gate-config", type=Path, default=Path("configs/phase3/gates.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gate_type == "audit":
        result = audit_gate(args.root, args.output)
    else:
        if args.aggregate is None:
            raise SystemExit("--aggregate is required for the primary gate")
        result = primary_gate(args.aggregate, args.output, args.gate_config)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result.get("pass", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
