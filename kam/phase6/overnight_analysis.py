"""Gates, aggregation, and reports for the Phase 6 overnight campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .overnight_manifest import (
    PREFLIGHT_ROWS,
    WAVE1_ROWS,
    WAVE2_ROWS,
    WAVE3_ROWS,
    build_wave2_rows,
    build_wave3_rows,
    read_jsonl,
    write_manifest,
)
from .stats import bootstrap_ci, equivalence_test, exact_permutation_test, holm_adjust, paired_effect


FINAL_OUTCOMES = (
    "PROMOTE_SPARSE_KAM_MEMORY",
    "PROMOTE_FIXED_KEY_FAST_ALGEBRA",
    "PROMOTE_KAM_FOR_ONLINE_ADAPTATION_ONLY",
    "PROMOTE_CONVENTIONAL_MEMORY_BASELINE",
    "PROMOTE_WIDENED_TRANSFORMER",
    "RETAIN_AS_DIAGNOSTIC_ONLY",
    "STOP_KAM_SPECIFIC_DIRECTION",
)


def _json_rows(root: Path, wave: str) -> list[dict[str, Any]]:
    directory = root / "rows" / wave
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if key == "metrics":
            continue
        flat[key] = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else value
    for key, value in row.get("metrics", {}).items():
        flat[f"metric_{key}"] = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else value
    return flat


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in materialized),
        encoding="utf-8",
    )


def _write_parquet(path: Path, rows: Iterable[dict[str, Any]], *, require: bool = True) -> dict[str, Any]:
    materialized = [_flatten(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path.with_suffix(".jsonl"), materialized)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(materialized) if materialized else pa.table({"empty": pa.array([], type=pa.string())})
        pq.write_table(table, path)
        return {"path": str(path), "rows": len(materialized), "engine": "pyarrow"}
    except Exception as exc:  # noqa: BLE001 - local environment may omit parquet
        if require:
            raise RuntimeError(f"Parquet export failed for {path}: {type(exc).__name__}: {exc}") from exc
        return {"path": str(path.with_suffix(".jsonl")), "rows": len(materialized), "engine": "jsonl_fallback"}


def _metric(row: dict[str, Any], name: str, default: float = math.inf) -> float:
    value = row.get("metrics", {}).get(name, row.get(f"metric_{name}", row.get(name, default)))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _gate(
    rows: list[dict[str, Any]],
    expected: int,
    *,
    wave: str,
    expected_ids: set[str] | None = None,
) -> dict[str, Any]:
    ids = [str(row.get("row_id")) for row in rows]
    failures = [str(row.get("row_id")) for row in rows if row.get("status") != "pass"]
    nonfinite = [str(row.get("row_id")) for row in rows if not _finite(row.get("metrics", {}))]
    duplicates = sorted({row_id for row_id in ids if ids.count(row_id) > 1})
    smoke = [str(row.get("row_id")) for row in rows if bool(row.get("metrics", {}).get("smoke_override"))]
    observed_ids = set(ids)
    missing_ids = sorted((expected_ids or set()) - observed_ids)
    unexpected_ids = sorted(observed_ids - (expected_ids or observed_ids))
    result = {
        "wave": wave,
        "pass": len(rows) == expected and not failures and not nonfinite and not duplicates and not smoke and not missing_ids and not unexpected_ids,
        "expected_rows": expected,
        "observed_rows": len(rows),
        "failure_row_ids": failures,
        "nonfinite_row_ids": nonfinite,
        "duplicate_row_ids": duplicates,
        "smoke_row_ids": smoke,
        "missing_manifest_row_ids": missing_ids,
        "unexpected_output_row_ids": unexpected_ids,
        "status_counts": dict(Counter(str(row.get("status", "missing")) for row in rows)),
    }
    return result


def preflight_gate(run_root: Path) -> dict[str, Any]:
    rows = _json_rows(run_root, "preflight")
    manifest = read_jsonl(run_root / "manifests" / "preflight.jsonl")
    result = _gate(rows, PREFLIGHT_ROWS, wave="preflight", expected_ids={str(row["row_id"]) for row in manifest})
    wrong_gpu = [
        str(row.get("row_id"))
        for row in rows
        if "L4" not in str(row.get("metadata", {}).get("gpu_name", "")).upper()
    ]
    data_failures: list[str] = []
    for row in rows:
        metrics = row.get("metrics", {})
        if row.get("lane") == "language" and (
            not metrics.get("dataset_sha256")
            or not metrics.get("tokenizer_sha256")
            or metrics.get("split_overlap") is not False
        ):
            data_failures.append(str(row.get("row_id")))
        if row.get("lane") == "dynamics" and (
            _metric(row, "finite_fraction", 0.0) != 1.0
            or _metric(row, "target_variance", 0.0) <= 1e-8
            or _metric(row, "clip_boundary_fraction", 1.0) >= 0.05
            or not metrics.get("nonconstant_stream")
        ):
            data_failures.append(str(row.get("row_id")))
    result["wrong_gpu_row_ids"] = wrong_gpu
    result["data_quality_failure_row_ids"] = sorted(set(data_failures))
    result["pass"] = bool(result["pass"] and not wrong_gpu and not data_failures)
    rates: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics", {})
        unit = "tokens" if row.get("lane") == "language" else "samples"
        rate = metrics.get(f"{unit}_per_second")
        if isinstance(rate, (int, float)) and float(rate) > 0:
            rates[f"{row.get('lane')}:{row.get('architecture')}"].append(float(rate))
            rates[f"{unit}:default"].append(float(rate))
    result["rates"] = {key: statistics.median(values) for key, values in rates.items()}
    destination = run_root / "calibration.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["pass"]:
        raise RuntimeError(f"preflight gate failed; see {destination}")
    return result


def _pareto(rows: list[dict[str, Any]], quality: str = "validation_loss") -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("status") == "pass" and math.isfinite(_metric(row, quality))]
    frontier: list[dict[str, Any]] = []
    for candidate in valid:
        objectives = (
            _metric(candidate, quality),
            _metric(candidate, "estimated_active_flops_per_token", _metric(candidate, "active_parameters_per_token")),
            _metric(candidate, "peak_vram_bytes"),
            _metric(candidate, "wall_seconds", _metric(candidate, "row_wall_seconds")),
        )
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            comparison = (
                _metric(other, quality),
                _metric(other, "estimated_active_flops_per_token", _metric(other, "active_parameters_per_token")),
                _metric(other, "peak_vram_bytes"),
                _metric(other, "wall_seconds", _metric(other, "row_wall_seconds")),
            )
            if all(left <= right for left, right in zip(comparison, objectives)) and any(
                left < right for left, right in zip(comparison, objectives)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def stage1_frontier(source: Path, run_root: Path, report_root: Path) -> dict[str, Any]:
    rows = read_jsonl(source)
    finite = [row for row in rows if row.get("status") == "pass" and _finite(row.get("metrics", {}))]
    if len(rows) < 3000:
        raise RuntimeError(f"completed Stage 1 frontier requires 3000 rows, found {len(rows)}")
    pareto = _pareto(finite, quality="loss")
    root = run_root
    _write_parquet(root / "stage1_frontier.parquet", finite)
    _write_parquet(root / "stage1_pareto.parquet", pareto)
    families = {
        "fixed": {"T-KAM-F"},
        "learned": {"T-KAM-L"},
        "alt": {"T-KAM-ALT"},
        "vp": {"T-KAM-VP"},
    }
    limits = {"fixed": 2, "learned": 2, "alt": 1, "vp": 1}
    selected: list[dict[str, Any]] = []
    for family, architectures in families.items():
        candidates = [
            row
            for row in pareto
            if str(row.get("architecture")) in architectures
            and _metric(row, "recall_at_k_against_exact", 1.0) >= 0.95
            and (family not in {"alt", "vp"} or _metric(row, "alternating_geometry_steps", _metric(row, "geometry_update_steps", 1.0)) > 0)
        ]
        selected.extend(sorted(candidates, key=lambda row: _metric(row, "loss"))[: limits[family]])
    selection = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_rows": len(rows),
        "finite_rows": len(finite),
        "pareto_rows": len(pareto),
        "selected_row_ids": [row.get("row_id") for row in selected],
        "selected": selected,
    }
    (root / "stage1_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True, default=str) + "\n")
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "STAGE1_FRONTIER_REANALYSIS.md").write_text(
        "# Stage 1 Frontier Reanalysis\n\n"
        f"- Source rows: {len(rows)}\n"
        f"- Finite valid rows: {len(finite)}\n"
        f"- Pareto rows: {len(pareto)}\n"
        f"- Selected configurations: {len(selected)} (maximum 6)\n\n"
        "The CPU reanalysis filters failed/nonfinite identities, router recall below 0.95, "
        "and ALT/VP rows without geometry updates before Pareto selection. The full selected "
        "records are in `results/phase6/overnight/stage1_selection.json`.\n",
        encoding="utf-8",
    )
    return selection


def aggregate_wave(run_root: Path, wave: str, *, report_root: Path) -> dict[str, Any]:
    expected = {"wave1": WAVE1_ROWS, "wave2": WAVE2_ROWS}[wave]
    rows = _json_rows(run_root, wave)
    manifest = read_jsonl(run_root / "manifests" / f"{wave}.jsonl")
    gate = _gate(rows, expected, wave=wave, expected_ids={str(row["row_id"]) for row in manifest})
    pareto = _pareto(rows)
    _write_parquet(run_root / f"{wave}_metrics.parquet", rows)
    _write_parquet(run_root / f"{wave}_pareto.parquet", pareto)
    (run_root / f"{wave}_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    if not gate["pass"]:
        raise RuntimeError(f"{wave} gate failed")
    if wave == "wave1":
        manifest = write_manifest(build_wave2_rows(rows), run_root / "manifests" / "wave2.jsonl")
    else:
        manifest = write_manifest(build_wave3_rows(rows), run_root / "manifests" / "wave3.jsonl")
    gate["next_manifest"] = manifest
    (run_root / f"{wave}_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    return gate


def validate_manifest(path: Path, expected: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    ids = [row["row_id"] for row in rows]
    result = {
        "path": str(path),
        "rows": len(rows),
        "expected": expected,
        "unique": len(ids) == len(set(ids)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if len(rows) != expected or not result["unique"]:
        raise RuntimeError(f"invalid generated manifest: {result}")
    return result


def _group_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _metric(row, metric)
        if row.get("status") == "pass" and math.isfinite(value):
            groups[str(row.get("architecture"))].append(value)
    return groups


LANGUAGE_LANES = frozenset({"language", "language_replication"})
CONVENTIONAL_MEMORY = frozenset({"T-MEMTOK", "T-MOE", "T-PKM"})
MIN_CONFIRMATORY_SEEDS = 6
ADAPTATION_METRICS = (
    "heldout_nmse",
    "early_post_transition_loss",
    "late_post_transition_loss",
    "cumulative_excess_loss",
    "recovery_time_steps",
    "reacquisition_time_steps",
    "update_flops",
    "adapter_state_bytes",
)


def _registered_token_target(row: dict[str, Any], subrun: dict[str, Any]) -> int | None:
    for source, key in (
        (row, "minimum_tokens_per_seed"),
        (row, "minimum_tokens"),
        (subrun, "target_tokens_resolved"),
    ):
        value = source.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
            return int(value)
    return None


def _matched_token_point(row: dict[str, Any], subrun: dict[str, Any]) -> tuple[float, float, int | None]:
    target = _registered_token_target(row, subrun)
    history = [
        point
        for point in subrun.get("loss_history", [])
        if isinstance(point.get("tokens"), (int, float))
        and isinstance(point.get("validation_loss"), (int, float))
        and math.isfinite(float(point["tokens"]))
        and math.isfinite(float(point["validation_loss"]))
    ]
    if not history:
        return _metric(subrun, "validation_loss"), _metric(subrun, "tokens", math.nan), target
    history = sorted(history, key=lambda point: float(point["tokens"]))
    if target is None:
        point = history[-1]
        return float(point["validation_loss"]), float(point["tokens"]), target
    before = [point for point in history if float(point["tokens"]) <= target]
    after = [point for point in history if float(point["tokens"]) >= target]
    left = before[-1] if before else history[0]
    right = after[0] if after else history[-1]
    left_tokens, right_tokens = float(left["tokens"]), float(right["tokens"])
    if right_tokens <= left_tokens:
        return float(left["validation_loss"]), left_tokens, target
    fraction = min(1.0, max(0.0, (target - left_tokens) / (right_tokens - left_tokens)))
    interpolated = float(left["validation_loss"]) + fraction * (
        float(right["validation_loss"]) - float(left["validation_loss"])
    )
    return interpolated, float(target), target


def _language_seed_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "pass" or str(row.get("lane")) not in LANGUAGE_LANES:
            continue
        subruns = row.get("metrics", {}).get("subruns", []) or [row.get("metrics", {})]
        for subrun in subruns:
            seed = subrun.get("training_seed", row.get("seed"))
            if not isinstance(seed, (int, float)):
                continue
            matched_loss, matched_tokens, target_tokens = _matched_token_point(row, subrun)
            observation = {
                "status": "pass",
                "wave": str(row.get("wave")),
                "lane": str(row.get("lane")),
                "task": str(row.get("task")),
                "scale": str(row.get("scale")),
                "architecture": str(row.get("architecture")),
                "training_seed": int(seed),
                "target_tokens": target_tokens,
                "matched_tokens": matched_tokens,
                "matched_token_validation_loss": matched_loss,
                "best_validation_loss": _metric(subrun, "best_validation_loss"),
                "final_validation_loss": _metric(subrun, "validation_loss"),
                "tokens": _metric(subrun, "tokens", math.nan),
                "wall_seconds": _metric(subrun, "wall_seconds", math.nan),
                "estimated_active_flops_per_token": _metric(subrun, "estimated_active_flops_per_token", math.nan),
                "active_parameters_per_token": _metric(subrun, "active_parameters_per_token", math.nan),
                "git_commit": row.get("metadata", {}).get("git_commit"),
                "git_dirty": bool(row.get("metadata", {}).get("git_dirty", False)),
                "row_id": row.get("row_id"),
                "history": subrun.get("loss_history", []),
            }
            if math.isfinite(matched_loss):
                observations.append(observation)
    return observations


def _dynamics_seed_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in rows:
        lane = str(row.get("lane"))
        if row.get("status") != "pass" or "dynamics" not in lane:
            continue
        subruns = row.get("metrics", {}).get("subruns", []) or [row.get("metrics", {})]
        for subrun in subruns:
            seed = subrun.get("training_seed", row.get("seed"))
            value = _metric(subrun, "heldout_nmse")
            if not isinstance(seed, (int, float)) or not math.isfinite(value):
                continue
            observations.append(
                {
                    "wave": str(row.get("wave")),
                    "lane": lane,
                    "task": str(subrun.get("task", row.get("task"))),
                    "scale": str(row.get("scale")),
                    "architecture": str(row.get("architecture")),
                    "training_seed": int(seed),
                    "heldout_nmse": value,
                    "row_id": row.get("row_id"),
                }
            )
    return observations


def _adaptation_base_seed(row: dict[str, Any], subrun: dict[str, Any]) -> int | None:
    seed = subrun.get("training_seed")
    if not isinstance(seed, (int, float)):
        return None
    schedule = subrun.get("schedule_index", 0)
    if isinstance(schedule, (int, float)):
        candidate = int(seed) - 10_000 * int(schedule)
        registered = {int(value) for value in row.get("seed_bundle", []) if isinstance(value, (int, float))}
        if not registered or candidate in registered:
            return candidate
    return int(seed) % 10_000


def _adaptation_seed_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    provenance: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "pass" or row.get("lane") != "adaptation":
            continue
        metrics = row.get("metrics", {})
        effective = str(metrics.get("adapter_effective", "joint_sgd_full_model"))
        declared = str(metrics.get("adapter_declared", row.get("adapter", "unknown")))
        registered = bool(metrics.get("adapter_registered", row.get("adapter_registered", False)))
        for subrun in metrics.get("subruns", []):
            base_seed = _adaptation_base_seed(row, subrun)
            if base_seed is None:
                continue
            key = (str(row.get("wave")), str(row.get("architecture")), effective, base_seed)
            grouped[key].append(subrun)
            provenance[key] = {
                "adapter_declared": declared,
                "adapter_registered": registered,
                "row_id": row.get("row_id"),
            }
    output: list[dict[str, Any]] = []
    for (wave, architecture, adapter, seed), subruns in sorted(grouped.items()):
        record: dict[str, Any] = {
            "wave": wave,
            "lane": "adaptation",
            "task": "registered_schedule_bundle",
            "architecture": architecture,
            "adapter_effective": adapter,
            "training_seed": seed,
            "subrun_count": len(subruns),
            "task_count": len({str(run.get("task")) for run in subruns}),
            "schedule_count": len({int(run.get("schedule_index", 0)) for run in subruns}),
            **provenance[(wave, architecture, adapter, seed)],
        }
        for metric in ADAPTATION_METRICS:
            values = [_metric(run, metric) for run in subruns]
            finite = [value for value in values if math.isfinite(value)]
            if finite:
                record[metric] = statistics.mean(finite)
        output.append(record)
    return output


def _candidate_pairs(observations: list[dict[str, Any]], metric: str) -> list[tuple[str, str, str]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in observations:
        value = _metric(row, metric)
        if math.isfinite(value):
            grouped[str(row.get("architecture"))].append(value)
    means = {name: statistics.mean(values) for name, values in grouped.items() if values}
    kam = sorted((name for name in means if name.startswith("T-KAM")), key=lambda name: means[name])
    conventional = sorted((name for name in means if name in CONVENTIONAL_MEMORY), key=lambda name: means[name])
    pairs: list[tuple[str, str, str]] = []
    if kam:
        best_kam = kam[0]
        for comparator in ("T0", "T-WIDE", conventional[0] if conventional else None):
            if comparator and comparator in means and comparator != best_kam:
                pairs.append((best_kam, comparator, "best_kam_primary"))
        joint = next((name for name in ("T-KAM-F", "T-KAM-L") if name in means), None)
        if joint:
            for architecture in ("T-KAM-ALT", "T-KAM-VP"):
                if architecture in means:
                    pairs.append((architecture, joint, "optimization_primary"))
    if "T-WIDE" in means and "T0" in means:
        pairs.append(("T-WIDE", "T0", "control_primary"))
    if conventional and "T0" in means:
        pairs.append((conventional[0], "T0", "conventional_primary"))
    unique: list[tuple[str, str, str]] = []
    for pair in pairs:
        if pair not in unique:
            unique.append(pair)
    return unique


def _paired_record(
    observations: list[dict[str, Any]],
    *,
    metric: str,
    architecture: str,
    comparator: str,
    comparison_family: str,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    candidate = {
        int(row["training_seed"]): float(row[metric])
        for row in observations
        if row.get("architecture") == architecture and math.isfinite(_metric(row, metric))
    }
    baseline = {
        int(row["training_seed"]): float(row[metric])
        for row in observations
        if row.get("architecture") == comparator and math.isfinite(_metric(row, metric))
    }
    seeds = sorted(candidate.keys() & baseline.keys())
    if len(seeds) < 2:
        return None
    left = [candidate[seed] for seed in seeds]
    right = [baseline[seed] for seed in seeds]
    differences = [value - reference for value, reference in zip(left, right)]
    effect = paired_effect(right, left)
    ci = bootstrap_ci(differences)
    permutation = exact_permutation_test(right, left)
    margin_fraction = 0.02 if "validation_loss" in metric else 0.05
    return {
        **context,
        "metric": metric,
        "architecture": architecture,
        "comparator": comparator,
        "comparison_family": comparison_family,
        "paired_seeds": len(seeds),
        "seed_ids": seeds,
        "architecture_mean": statistics.mean(left),
        "comparator_mean": statistics.mean(right),
        "mean_difference": statistics.mean(differences),
        "relative_difference": statistics.mean(differences) / max(abs(statistics.mean(right)), 1e-12),
        "bootstrap_ci_low": ci[0],
        "bootstrap_ci_high": ci[1],
        "standardized_paired_effect": effect["effect_size_dz"],
        "exact_paired_permutation_p": permutation["p_value"],
        "holm_adjusted_p": permutation["p_value"],
        "equivalent": equivalence_test(
            right,
            left,
            margin=margin_fraction * max(abs(statistics.mean(right)), 1e-12),
        ).get("equivalent"),
    }


def _paired_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    language_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in _language_seed_observations(rows):
        language_groups[(row["wave"], row["lane"], row["task"], row["scale"], row["target_tokens"])].append(row)
    for (wave, lane, task, scale, target_tokens), group in language_groups.items():
        context = {"wave": wave, "lane": lane, "task": task, "scale": scale, "target_tokens": target_tokens}
        for metric in ("matched_token_validation_loss", "best_validation_loss", "final_validation_loss"):
            for architecture, comparator, family in _candidate_pairs(group, metric):
                record = _paired_record(
                    group,
                    metric=metric,
                    architecture=architecture,
                    comparator=comparator,
                    comparison_family=family,
                    context=context,
                )
                if record:
                    output.append(record)

    dynamics_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in _dynamics_seed_observations(rows):
        dynamics_groups[(row["wave"], row["lane"], row["task"], row["scale"])].append(row)
    for (wave, lane, task, scale), group in dynamics_groups.items():
        context = {"wave": wave, "lane": lane, "task": task, "scale": scale, "target_tokens": None}
        for architecture, comparator, family in _candidate_pairs(group, "heldout_nmse"):
            record = _paired_record(
                group,
                metric="heldout_nmse",
                architecture=architecture,
                comparator=comparator,
                comparison_family=family,
                context=context,
            )
            if record:
                output.append(record)

    adaptation_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _adaptation_seed_observations(rows):
        adaptation_groups[(str(row["wave"]), str(row["adapter_effective"]))].append(row)
    for (wave, adapter), group in adaptation_groups.items():
        registered = all(bool(row.get("adapter_registered")) for row in group)
        context = {
            "wave": wave,
            "lane": "adaptation",
            "task": "registered_schedule_bundle",
            "scale": "2M",
            "target_tokens": None,
            "adapter_effective": adapter,
            "adapter_registered": registered,
        }
        for metric in ADAPTATION_METRICS:
            for architecture, comparator, family in _candidate_pairs(group, metric):
                record = _paired_record(
                    group,
                    metric=metric,
                    architecture=architecture,
                    comparator=comparator,
                    comparison_family=(family if registered else "exploratory_unregistered_full_model"),
                    context=context,
                )
                if record:
                    output.append(record)

    families: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        family = (row.get("wave"), row.get("lane"), row.get("task"), row.get("metric"), row.get("comparison_family"))
        families[family].append(index)
    for indices in families.values():
        adjusted = holm_adjust({str(index): float(output[index]["exact_paired_permutation_p"]) for index in indices})
        for index in indices:
            output[index]["holm_adjusted_p"] = adjusted[str(index)]
    return output


def _decision(rows: list[dict[str, Any]], paired: list[dict[str, Any]] | None = None) -> tuple[str, str]:
    paired = paired if paired is not None else _paired_statistics(rows)
    replication = [row for row in _language_seed_observations(rows) if row["wave"] == "wave3"]
    means = {
        name: statistics.mean(values)
        for name, values in _group_metric(replication, "matched_token_validation_loss").items()
        if values
    }
    kam = sorted((name for name in means if name.startswith("T-KAM")), key=lambda name: means[name])

    def comparison(architecture: str, comparator: str) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in paired
                if row.get("wave") == "wave3"
                and row.get("lane") == "language_replication"
                and row.get("metric") == "matched_token_validation_loss"
                and row.get("architecture") == architecture
                and row.get("comparator") == comparator
            ),
            None,
        )

    def confirmatory(record: dict[str, Any] | None) -> bool:
        return bool(
            record
            and int(record["paired_seeds"]) >= MIN_CONFIRMATORY_SEEDS
            and float(record["holm_adjusted_p"]) <= 0.05
            and float(record["bootstrap_ci_high"]) < 0.0
        )

    if kam and "T-WIDE" in means:
        best_kam = kam[0]
        versus_wide = comparison(best_kam, "T-WIDE")
        if versus_wide and float(versus_wide["relative_difference"]) <= -0.02 and confirmatory(versus_wide):
            outcome = "PROMOTE_FIXED_KEY_FAST_ALGEBRA" if best_kam == "T-KAM-F" else "PROMOTE_SPARSE_KAM_MEMORY"
            return outcome, (
                f"{best_kam} cleared the 2% Wave 3 matched-token threshold with "
                f"{versus_wide['paired_seeds']} paired seeds and Holm-adjusted p={versus_wide['holm_adjusted_p']:.4g}."
            )
        if versus_wide:
            direction = "better" if float(versus_wide["relative_difference"]) < 0 else "worse"
            return "RETAIN_AS_DIAGNOSTIC_ONLY", (
                f"{best_kam} was {abs(100 * float(versus_wide['relative_difference'])):.1f}% {direction} than T-WIDE "
                f"at the registered Wave 3 token checkpoint, but only {versus_wide['paired_seeds']} paired seeds were available "
                f"(exact p={versus_wide['exact_paired_permutation_p']:.4g}, Holm p={versus_wide['holm_adjusted_p']:.4g}); "
                "the confirmatory gate is underpowered."
            )

    registered_adaptation = [
        row
        for row in paired
        if row.get("lane") == "adaptation"
        and row.get("metric") == "late_post_transition_loss"
        and row.get("adapter_registered") is True
    ]
    if any(float(row["relative_difference"]) <= -0.05 and confirmatory(row) for row in registered_adaptation):
        return "PROMOTE_KAM_FOR_ONLINE_ADAPTATION_ONLY", "A KAM cleared the registered matched-adapter confirmation gate."
    return "RETAIN_AS_DIAGNOSTIC_ONLY", (
        "No Wave 3 architecture cleared a seed-level, Holm-corrected confirmation gate. "
        "The adaptation lane used unregistered full-model updates and cannot support an adapter promotion."
    )


def _report_header(title: str, rows: list[dict[str, Any]]) -> str:
    return (
        f"# {title}\n\n"
        f"Campaign rows analyzed: {len(rows)}. Status counts: "
        f"`{dict(Counter(str(row.get('status')) for row in rows))}`.\n\n"
    )


def _build_figures(rows: list[dict[str, Any]], report_root: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    colors = {
        "T0": "#64748b",
        "T-WIDE": "#2563eb",
        "T-MEMTOK": "#c2410c",
        "T-MOE": "#a16207",
        "T-PKM": "#7c3aed",
        "T-KAM-F": "#be123c",
        "T-KAM-L": "#9f1239",
        "T-KAM-ALT": "#db2777",
        "T-KAM-VP": "#475569",
    }

    def finish(figure, name: str) -> None:
        path = figure_root / name
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        created.append(str(path))

    language = _language_seed_observations(rows)
    waves = [wave for wave in ("wave1", "wave2", "wave3") if any(row["wave"] == wave for row in language)]
    figure, axes = plt.subplots(len(waves) or 1, 1, figsize=(10, 3.8 * max(len(waves), 1)), squeeze=False)
    for axis, wave in zip(axes[:, 0], waves):
        selected = [row for row in language if row["wave"] == wave]
        target = min((int(row["target_tokens"]) for row in selected if row.get("target_tokens")), default=None)
        for architecture in sorted({row["architecture"] for row in selected}):
            architecture_rows = [row for row in selected if row["architecture"] == architecture]
            interpolated = []
            start_tokens = max((float(row["history"][0].get("tokens", 0.0)) for row in architecture_rows if row["history"]), default=0.0)
            grid = np.linspace(start_tokens, float(target), 60) if target and start_tokens < target else None
            for row in architecture_rows:
                points = [
                    point
                    for point in row["history"]
                    if isinstance(point.get("tokens"), (int, float))
                    and isinstance(point.get("validation_loss"), (int, float))
                    and (target is None or float(point["tokens"]) <= 1.05 * target)
                ]
                if len(points) < 2:
                    continue
                x = np.asarray([float(point["tokens"]) / 1e6 for point in points])
                y = np.asarray([float(point["validation_loss"]) for point in points])
                axis.plot(x, y, color=colors.get(architecture, "#334155"), alpha=0.13, linewidth=0.8)
                if grid is not None:
                    clipped = grid[(grid >= float(points[0]["tokens"])) & (grid <= float(points[-1]["tokens"]))]
                    if clipped.size == grid.size:
                        interpolated.append(np.interp(grid, np.asarray([float(point["tokens"]) for point in points]), y))
            if interpolated and grid is not None:
                values = np.asarray(interpolated)
                mean = values.mean(axis=0)
                standard_error = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]) if values.shape[0] > 1 else np.zeros_like(mean)
                axis.plot(grid / 1e6, mean, color=colors.get(architecture, "#334155"), linewidth=2.0, label=f"{architecture} (n={values.shape[0]})")
                axis.fill_between(grid / 1e6, mean - 1.96 * standard_error, mean + 1.96 * standard_error, color=colors.get(architecture, "#334155"), alpha=0.10)
        axis.set_title(f"{wave.capitalize()} language validation learning curves")
        axis.set_xlabel("Training tokens (millions)")
        axis.set_ylabel("Validation cross-entropy")
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.7)
        if axis.get_legend_handles_labels()[1]:
            axis.legend(fontsize=7, ncol=3)
    if not waves:
        axes[0, 0].text(0.5, 0.5, "No completed language trajectories", ha="center", va="center")
    finish(figure, "language_learning_curves_by_wave.png")

    wave3 = [row for row in language if row["wave"] == "wave3"]
    figure, axis = plt.subplots(figsize=(9, 5))
    architectures = sorted({row["architecture"] for row in wave3})
    positions = np.arange(len(architectures))
    width = 0.24
    for offset, (metric, label) in enumerate(
        (
            ("best_validation_loss", "best checkpoint"),
            ("matched_token_validation_loss", "registered token checkpoint"),
            ("final_validation_loss", "wall-time final"),
        )
    ):
        means = [statistics.mean(float(row[metric]) for row in wave3 if row["architecture"] == architecture) for architecture in architectures]
        axis.bar(positions + (offset - 1) * width, means, width=width, label=label, alpha=0.88)
    if architectures:
        axis.set_xticks(positions, architectures, rotation=20, ha="right")
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "No completed Wave 3 language rows", ha="center", va="center")
    axis.set_title("Wave 3 language loss by checkpoint policy")
    axis.set_ylabel("Validation cross-entropy")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    finish(figure, "language_checkpoint_policy_comparison.png")

    trace_groups: dict[tuple[str, int], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row.get("wave") != "wave2" or row.get("lane") != "dynamics_bundle":
            continue
        for subrun in row.get("metrics", {}).get("subruns", []):
            seed = subrun.get("training_seed")
            if isinstance(seed, (int, float)) and subrun.get("prediction_trace") and subrun.get("truth_trace"):
                trace_groups[(str(subrun.get("task")), int(seed))].append((str(row.get("architecture")), subrun))
    representative = max(sorted(trace_groups), key=lambda key: len(trace_groups[key])) if trace_groups else None
    traces = sorted(trace_groups.get(representative, []))
    figure, axes = plt.subplots(max(len(traces), 1), 2, figsize=(12, 2.8 * max(len(traces), 1)), squeeze=False)
    for row_index, (architecture, subrun) in enumerate(traces):
        truth = np.asarray(subrun["truth_trace"], dtype=float)
        prediction = np.asarray(subrun["prediction_trace"], dtype=float)
        count = min(truth.size, prediction.size, 256)
        error = np.abs(prediction[:count] - truth[:count])
        axes[row_index, 0].plot(truth[:count], color="#334155", linewidth=1.2, label="true")
        axes[row_index, 0].plot(prediction[:count], color=colors.get(architecture, "#be123c"), linewidth=1.0, alpha=0.85, label="prediction")
        axes[row_index, 0].set_ylabel(architecture)
        axes[row_index, 0].legend(fontsize=7, ncol=2)
        axes[row_index, 1].semilogy(np.maximum(error, 1e-8), color=colors.get(architecture, "#be123c"), linewidth=1.0)
        axes[row_index, 1].set_ylabel("absolute error")
    if representative:
        axes[0, 0].set_title(f"Prediction and truth: {representative[0]}, seed {representative[1]}")
        axes[0, 1].set_title("Absolute error (log scale)")
        axes[-1, 0].set_xlabel("Held-out timestep")
        axes[-1, 1].set_xlabel("Held-out timestep")
    else:
        axes[0, 0].text(0.5, 0.5, "No comparable dynamics traces", ha="center", va="center")
    finish(figure, "dynamics_prediction_true_error_comparable.png")

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    memory_rows = [
        row
        for row in rows
        if row.get("wave") == "wave1"
        and row.get("lane") == "language"
        and row.get("architecture") in {"T-KAM-L", "T-KAM-ALT", "T-KAM-VP"}
    ]
    seen_labels: set[str] = set()
    for row in memory_rows:
        architecture = str(row.get("architecture"))
        for subrun in row.get("metrics", {}).get("subruns", []):
            history = subrun.get("loss_history", [])
            if not history:
                continue
            steps = [float(point.get("step", index)) for index, point in enumerate(history)]
            label = architecture if architecture not in seen_labels else None
            axes[0].plot(steps, [float(point.get("memory_gate_mean", 0.0)) for point in history], color=colors.get(architecture), alpha=0.75, label=label)
            axes[1].semilogy(steps, [max(float(point.get("memory_key_grad_norm", 0.0)), 1e-12) for point in history], color=colors.get(architecture), alpha=0.75, label=label)
            freeze = subrun.get("geometry_freeze_step")
            if isinstance(freeze, (int, float)):
                for axis in axes:
                    axis.axvline(float(freeze), color=colors.get(architecture), linestyle="--", alpha=0.25)
            seen_labels.add(architecture)
    axes[0].set_title("Learned-memory adaptation and final-tuning freeze (Wave 1)")
    axes[0].set_ylabel("Memory gate scale")
    axes[1].set_ylabel("Key-gradient norm (log)")
    axes[1].set_xlabel("Optimizer step")
    if memory_rows:
        axes[0].legend(fontsize=8, ncol=3)
    else:
        axes[0].text(0.5, 0.5, "No learned-memory language trajectories", ha="center", va="center")
    finish(figure, "memory_adaptation_freeze_learned_variants.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    for architecture in architectures:
        selected = [row for row in wave3 if row["architecture"] == architecture]
        x = statistics.mean(float(row["active_parameters_per_token"]) for row in selected)
        y = statistics.mean(float(row["matched_token_validation_loss"]) for row in selected)
        axis.scatter(x, y, s=65, color=colors.get(architecture, "#334155"), edgecolor="white", linewidth=0.7)
        axis.annotate(architecture, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_title("Wave 3 matched-token quality and active parameters")
    axis.set_xlabel("Active parameters per token (log scale)")
    axis.set_ylabel("Validation cross-entropy at registered token checkpoint")
    if architectures:
        axis.set_xscale("log")
    axis.grid(color="#e2e8f0", linewidth=0.7)
    finish(figure, "resource_quality_wave3_matched.png")

    adaptation = _adaptation_seed_observations(rows)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = (
        ("late_post_transition_loss", "Late post-transition loss", False),
        ("cumulative_excess_loss", "Cumulative excess loss", False),
        ("recovery_time_steps", "Recovery time (steps)", False),
        ("update_flops", "Update FLOPs", True),
    )
    for axis, (metric, title, log_scale) in zip(axes.flat, metrics):
        available = [architecture for architecture in sorted({row["architecture"] for row in adaptation}) if any(math.isfinite(_metric(row, metric)) for row in adaptation if row["architecture"] == architecture)]
        for index, architecture in enumerate(available):
            values = [float(row[metric]) for row in adaptation if row["architecture"] == architecture and metric in row]
            axis.scatter([index] * len(values), values, color=colors.get(architecture, "#334155"), alpha=0.55, s=25)
            if values:
                axis.scatter(index, statistics.mean(values), color=colors.get(architecture, "#334155"), edgecolor="black", linewidth=0.6, s=80, marker="D")
        axis.set_xticks(range(len(available)), available, rotation=20, ha="right")
        axis.set_title(title)
        if log_scale and available:
            axis.set_yscale("log")
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    figure.suptitle("Online adaptation by base seed — effective method: full-model joint SGD (unregistered)")
    if not adaptation:
        axes[0, 0].text(0.5, 0.5, "No completed adaptation seed metrics", ha="center", va="center")
    finish(figure, "adaptation_seed_metrics.png")
    return created


def _language_report_detail(language: list[dict[str, Any]], paired: list[dict[str, Any]]) -> str:
    wave3 = [row for row in language if row["wave"] == "wave3"]
    lines = [
        "## Wave 3 checkpoint comparison",
        "",
        "| Architecture | Seeds | Best checkpoint | Registered-token checkpoint | Final checkpoint | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for architecture in sorted({row["architecture"] for row in wave3}):
        selected = [row for row in wave3 if row["architecture"] == architecture]
        lines.append(
            f"| {architecture} | {len(selected)} | "
            f"{statistics.mean(row['best_validation_loss'] for row in selected):.4f} | "
            f"{statistics.mean(row['matched_token_validation_loss'] for row in selected):.4f} | "
            f"{statistics.mean(row['final_validation_loss'] for row in selected):.4f} | "
            f"{statistics.mean(row['tokens'] for row in selected) / 1e6:.1f}M |"
        )
    lines.extend(
        [
            "",
            "The registered-token checkpoint is the primary quality basis. Best and final checkpoints are descriptive diagnostics; final checkpoints contain unequal token exposure from the completed legacy runs.",
            "",
            "## Registered Wave 3 paired comparisons",
            "",
            "| Candidate | Comparator | Seeds | Relative difference | Bootstrap 95% CI | Exact p | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    selected_pairs = [
        row
        for row in paired
        if row.get("wave") == "wave3"
        and row.get("lane") == "language_replication"
        and row.get("metric") == "matched_token_validation_loss"
    ]
    for row in selected_pairs:
        lines.append(
            f"| {row['architecture']} | {row['comparator']} | {row['paired_seeds']} | "
            f"{100 * row['relative_difference']:.1f}% | "
            f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}] | "
            f"{row['exact_paired_permutation_p']:.4g} | {row['holm_adjusted_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            f"Confirmatory promotion requires at least {MIN_CONFIRMATORY_SEEDS} paired seeds, a favorable bootstrap interval, and Holm-adjusted p ≤ 0.05.",
            "",
        ]
    )
    return "\n".join(lines)


def _adaptation_report_detail(adaptation: list[dict[str, Any]]) -> str:
    lines = [
        "## Seed-level descriptive metrics",
        "",
        "| Architecture | Seeds | Held-out NMSE | Late transition loss | Recovery steps | Update FLOPs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for architecture in sorted({row["architecture"] for row in adaptation}):
        selected = [row for row in adaptation if row["architecture"] == architecture]
        mean = lambda metric: statistics.mean(float(row[metric]) for row in selected if metric in row)
        lines.append(
            f"| {architecture} | {len(selected)} | {mean('heldout_nmse'):.4g} | "
            f"{mean('late_post_transition_loss'):.4g} | {mean('recovery_time_steps'):.3g} | {mean('update_flops'):.4g} |"
        )
    lines.extend(
        [
            "",
            "Each row is one base training seed after averaging its registered tasks and held-out schedules.",
            "",
            "Important: the completed overnight runner used full-model joint-SGD updates for every architecture. The manifest labels `rls` and `value_only` were declarations, not implemented adapter dispatch. These results are therefore exploratory architecture-adaptation diagnostics, not registered matched-adapter evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def final_aggregate(run_root: Path, report_root: Path) -> dict[str, Any]:
    wave3 = _json_rows(run_root, "wave3")
    manifest = read_jsonl(run_root / "manifests" / "wave3.jsonl")
    gate = _gate(wave3, WAVE3_ROWS, wave="wave3", expected_ids={str(row["row_id"]) for row in manifest})
    (run_root / "wave3_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    if not gate["pass"]:
        raise RuntimeError("wave3 gate failed")
    rows = sum((_json_rows(run_root, wave) for wave in ("preflight", "wave1", "wave2", "wave3")), [])
    manifests = sum(
        (read_jsonl(run_root / "manifests" / f"{wave}.jsonl") for wave in ("preflight", "wave1", "wave2", "wave3")),
        [],
    )
    paired = _paired_statistics(rows)
    language_seed_metrics = _language_seed_observations(rows)
    dynamics_seed_metrics = _dynamics_seed_observations(rows)
    adaptation_raw = [row for row in rows if row.get("lane") == "adaptation"]
    adaptation = _adaptation_seed_observations(rows)
    deletion: list[dict[str, Any]] = []
    for row in rows:
        for subrun in row.get("metrics", {}).get("subruns", []):
            for diagnostic in subrun.get("deletion_metrics", []):
                deletion.append({"row_id": row.get("row_id"), "architecture": row.get("architecture"), **diagnostic})
    resources = [
        {
            "row_id": row.get("row_id"),
            "wave": row.get("wave"),
            "architecture": row.get("architecture"),
            "gpu_name": row.get("metadata", {}).get("gpu_name"),
            "wall_seconds": _metric(row, "wall_seconds", row.get("row_wall_seconds", math.inf)),
            "peak_vram_bytes": _metric(row, "peak_vram_bytes", 0.0),
            "total_parameters": _metric(row, "total_parameters", 0.0),
            "active_parameters_per_token": _metric(row, "active_parameters_per_token", 0.0),
        }
        for row in rows
    ]
    failures = [row for row in rows if row.get("status") != "pass"]
    exports = {
        "run_manifest": _write_parquet(run_root / "run_manifest.parquet", manifests),
        "all_metrics": _write_parquet(run_root / "all_metrics.parquet", rows),
        "language_seed_metrics": _write_parquet(run_root / "language_seed_metrics.parquet", language_seed_metrics),
        "dynamics_seed_metrics": _write_parquet(run_root / "dynamics_seed_metrics.parquet", dynamics_seed_metrics),
        "paired_seed_metrics": _write_parquet(run_root / "paired_seed_metrics.parquet", paired),
        "adaptation_metrics": _write_parquet(run_root / "adaptation_metrics.parquet", adaptation),
        "adaptation_row_metrics": _write_parquet(run_root / "adaptation_row_metrics.parquet", adaptation_raw),
        "deletion_metrics": _write_parquet(run_root / "deletion_metrics.parquet", deletion),
        "resource_metrics": _write_parquet(run_root / "resource_metrics.parquet", resources),
        "failures": _write_parquet(run_root / "failures.parquet", failures),
    }
    outcome, rationale = _decision(rows, paired)
    if outcome not in FINAL_OUTCOMES:
        raise AssertionError("decision outside registered outcome set")
    report_root.mkdir(parents=True, exist_ok=True)
    figures = _build_figures(rows, report_root)
    language_rows = [row for row in rows if "language" in str(row.get("lane"))]
    dynamics_rows = [row for row in rows if "dynamics" in str(row.get("lane"))]
    reports = {
        "OVERNIGHT_EXECUTION_REPORT.md": _report_header("Phase 6 Overnight Execution Report", rows)
        + f"Final outcome: `{outcome}`.\n\n{rationale}\n\nMachine-readable exports: `{json.dumps(exports, sort_keys=True)}`.\n",
        "OVERNIGHT_LANGUAGE_REPORT.md": _report_header("Phase 6 Overnight Language Report", language_rows)
        + _language_report_detail(language_seed_metrics, paired),
        "OVERNIGHT_DYNAMICS_REPORT.md": _report_header("Phase 6 Overnight Dynamics Report", dynamics_rows)
        + "Dynamics rows report held-out NMSE, validation trajectories, optimizer phase counts, conditioning, and data-quality checks.\n",
        "OVERNIGHT_OPTIMIZATION_REPORT.md": _report_header("Phase 6 Overnight Optimization Report", rows)
        + "ALT rows expose algebra/geometry step counts; VP rows freeze geometry under stop-gradient. "
        "Stage 1 Pareto selection is documented in `STAGE1_FRONTIER_REANALYSIS.md`.\n",
        "OVERNIGHT_ADAPTATION_REPORT.md": _report_header("Phase 6 Overnight Adaptation Report", adaptation_raw)
        + _adaptation_report_detail(adaptation),
        "OVERNIGHT_DECISION_MEMO.md": "# Phase 6 Overnight Decision Memo\n\n"
        + f"## Decision\n\n`{outcome}`\n\n## Rationale\n\n{rationale}\n\n"
        + "The corrected gate uses only Wave 3, exact seed identity, the registered token checkpoint, bootstrap uncertainty, exact paired permutation tests, and within-family Holm correction. A completed run is not itself evidence for KAM.\n",
        "OVERNIGHT_REPRODUCIBILITY.md": "# Phase 6 Overnight Reproducibility\n\n"
        + "Each row records commit/dirty state, manifest hash, architecture, dataset/tokenizer checksums, seeds, precision, "
        "GPU and framework versions, parameter/resource accounting, budgets, throughput, checkpoints, and failure category. "
        "The immutable row manifests and Slurm dependency graph are under `results/phase6/overnight/`.\n",
    }
    for name, body in reports.items():
        (report_root / name).write_text(body, encoding="utf-8")
    summary = {
        "analysis_version": "phase6_overnight_v2_seed_stratified",
        "decision_basis": "wave3_registered_token_checkpoint",
        "gate": gate,
        "row_count": len(rows),
        "decision": outcome,
        "rationale": rationale,
        "exports": exports,
        "figures": figures,
    }
    (run_root / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight-gate", "final"):
        child = subparsers.add_parser(name)
        child.add_argument("--run-root", required=True)
        if name == "final":
            child.add_argument("--report-root", required=True)
    frontier = subparsers.add_parser("stage1-frontier")
    frontier.add_argument("--source", required=True)
    frontier.add_argument("--run-root", required=True)
    frontier.add_argument("--report-root", required=True)
    aggregate = subparsers.add_parser("aggregate-wave")
    aggregate.add_argument("--wave", choices=("wave1", "wave2"), required=True)
    aggregate.add_argument("--run-root", required=True)
    aggregate.add_argument("--report-root", required=True)
    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--path", required=True)
    validate.add_argument("--expected", required=True, type=int)
    args = parser.parse_args()
    run_root = Path(getattr(args, "run_root", "results/phase6/overnight"))
    if args.command == "preflight-gate":
        result = preflight_gate(run_root)
    elif args.command == "stage1-frontier":
        result = stage1_frontier(Path(args.source), run_root, Path(args.report_root))
    elif args.command == "aggregate-wave":
        result = aggregate_wave(run_root, args.wave, report_root=Path(args.report_root))
    elif args.command == "validate-manifest":
        result = validate_manifest(Path(args.path), args.expected)
    else:
        result = final_aggregate(run_root, Path(args.report_root))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
