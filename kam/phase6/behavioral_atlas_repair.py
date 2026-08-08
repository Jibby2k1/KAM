"""Revisioned Stage 0 measurement repair for the Phase 6.2 behavioral atlas.

The original Stage 0 manifest and report remain immutable. This module creates
and evaluates a small repair manifest, then merges those results with the
original non-compile evidence under explicitly versioned decision rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any

import torch

from kam.phase6.behavioral_atlas_analysis import audit_results, load_results, profile_metrics
from kam.phase6.behavioral_atlas_instrumentation import build_anchor_bank, evaluate_anchor_behavior
from kam.phase6.behavioral_atlas_manifest import build_behavioral_atlas_rows
from kam.phase6.behavioral_atlas_runner import build_behavioral_atlas_model, run_behavioral_atlas_row
from kam.phase6.overnight_runner import _language_corpus

REPAIR_VERSION = "stage0_measurement_repair_r2"
BF16_TOP1_TOLERANCE = 0.02
BF16_PREDICTIVE_KL_TOLERANCE = 1e-3
ANCHOR_RELATIVE_TOLERANCE = 0.05
ANCHOR_STANDARD_STATES = 16_384
ANCHOR_DOUBLED_STATES = 32_768


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _repair_id(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "p6atlas_repair_" + hashlib.sha256(data).hexdigest()[:16]


def build_repair_rows(stage0_manifest: str | Path | None = None) -> list[dict[str, Any]]:
    if stage0_manifest is None:
        stage0 = build_behavioral_atlas_rows("stage0")
    else:
        stage0 = [
            json.loads(line)
            for line in Path(stage0_manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    source = next(
        row for row in stage0
        if row["profile_kind"] is None
        and row["arm"] == "learned_joint_freeze80"
        and row["seed"] == 76_001
    )
    failed_compile = next(row for row in stage0 if row["profile_kind"] == "compile_candidate")
    anchor = {
        "campaign": "phase6_behavioral_atlas_v2",
        "stage": REPAIR_VERSION,
        "repair_kind": "anchor_checkpoint_reevaluation",
        "source_row_id": source["row_id"],
        "standard_anchor_token_states": ANCHOR_STANDARD_STATES,
        "doubled_anchor_token_states": ANCHOR_DOUBLED_STATES,
        "inferential": False,
        "preregistered": True,
    }
    anchor["row_id"] = _repair_id(anchor)
    compile_row = {
        **failed_compile,
        "stage": REPAIR_VERSION,
        "repair_kind": "compile_candidate_no_cudagraph",
        "profile_kind": "compile_candidate_no_cudagraph",
        "supersedes_row_id": failed_compile["row_id"],
        "anchor_token_states": ANCHOR_STANDARD_STATES,
        "compile_training": True,
        "compile_mode": "default",
        "compile_cudagraphs": False,
        "inferential": False,
        "scientific_role": "noninferential_stage0_compile_repair",
    }
    compile_row.pop("row_id")
    compile_row["row_id"] = _repair_id(compile_row)
    return [anchor, compile_row]


def write_repair_manifest(path: str | Path, stage0_manifest: str | Path | None = None) -> dict[str, Any]:
    rows = build_repair_rows(stage0_manifest)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "repair_version": REPAIR_VERSION,
        "rows": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "path": str(destination),
        "inferential": False,
    }


def _relative_difference(first: float, second: float) -> float:
    return abs(float(first) - float(second)) / max(abs(float(second)), 1e-12)


def _anchor_view(behavior: dict[str, Any]) -> dict[str, float]:
    routing = behavior["routing_decomposition"]["states"]["Qt_Kt"]
    return {
        "memory_output_stable_rank_mean": float(behavior["memory_output_stable_rank_mean"]),
        "memory_output_participation_ratio_mean": float(behavior["memory_output_participation_ratio_mean"]),
        "memory_contribution_ratio_mean": float(behavior["memory_contribution_ratio_mean"]),
        "normalized_support_entropy": float(routing["normalized_support_entropy"]),
        "global_effective_support_count": float(routing["global_effective_support_count"]),
        "dead_support_fraction": float(routing["dead_support_fraction"]),
    }


def evaluate_anchor_checkpoint(
    spec: dict[str, Any], *, stage0_run_root: Path, output_root: Path, device: str
) -> dict[str, Any]:
    source_result_path = stage0_run_root / "rows" / "behavioral_atlas_v2" / f"{spec['source_row_id']}.json"
    source = json.loads(source_result_path.read_text(encoding="utf-8"))
    snapshot = Path(source["model_snapshots"][-1])
    if not snapshot.is_file():
        snapshot = stage0_run_root / "snapshots" / str(spec["source_row_id"]) / "model_2000000.pt"
    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    row = payload["row"]
    resolved_device = torch.device(device)
    model = build_behavioral_atlas_model(row).to(resolved_device)
    model.load_state_dict(payload["model"])
    model.eval()
    tokens, dataset = _language_corpus(row)
    validation_range = tuple(int(value) for value in dataset["validation_range"])
    behaviors: dict[str, dict[str, Any]] = {}
    for name, states in (
        ("standard", int(spec["standard_anchor_token_states"])),
        ("doubled", int(spec["doubled_anchor_token_states"])),
    ):
        bank = build_anchor_bank(
            tokens,
            validation_range,
            sequence_length=int(row["sequence_length"]),
            token_states=states,
            seed=int(row["anchor_seed"]),
        )
        behavior, _ = evaluate_anchor_behavior(
            model,
            bank,
            batch_size=int(row.get("anchor_batch_size", 16)),
            device=resolved_device,
            precision=str(row.get("precision", "bf16")),
            router_metric=str(row.get("router_metric", "dot")),
            router_temperature=float(row.get("router_temperature", 1.0)),
            top_k=int(row.get("top_k", 4)),
        )
        behaviors[name] = behavior
    standard = _anchor_view(behaviors["standard"])
    doubled = _anchor_view(behaviors["doubled"])
    invariant_fields = (
        "memory_output_stable_rank_mean",
        "memory_output_participation_ratio_mean",
        "memory_contribution_ratio_mean",
        "normalized_support_entropy",
        "global_effective_support_count",
    )
    differences = {field: _relative_difference(standard[field], doubled[field]) for field in invariant_fields}
    dead_layers = {
        name: [
            float(layer["Qt_Kt"]["dead_support_fraction"])
            for layer in behavior["routing_decomposition"]["per_layer"]
        ]
        for name, behavior in behaviors.items()
    }
    result = {
        **spec,
        "status": "pass",
        "source_result_path": str(source_result_path),
        "source_snapshot_path": str(snapshot),
        "anchor_gate_version": "fixed_bank_invariant_metrics_v2",
        "invariant_relative_tolerance": ANCHOR_RELATIVE_TOLERANCE,
        "invariant_relative_differences": differences,
        "anchor_sufficiency_pass": max(differences.values()) <= ANCHOR_RELATIVE_TOLERANCE,
        "standard_metrics": standard,
        "doubled_metrics": doubled,
        "dead_support_diagnostic": {
            "reason_not_in_invariance_gate": "unseen-support occupancy decreases mechanically as the sampled bank grows",
            "standard_mean": standard["dead_support_fraction"],
            "doubled_mean": doubled["dead_support_fraction"],
            "standard_layer_range": [min(dead_layers["standard"]), max(dead_layers["standard"])],
            "doubled_layer_range": [min(dead_layers["doubled"]), max(dead_layers["doubled"])],
        },
    }
    _atomic_json(output_root / "rows" / "repair" / f"{spec['row_id']}.json", result)
    return result


def run_repair_manifest_row(
    manifest: str | Path,
    index: int,
    *,
    stage0_run_root: str | Path,
    output_root: str | Path,
    device: str,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    spec = rows[index]
    output = Path(output_root)
    try:
        if spec["repair_kind"] == "anchor_checkpoint_reevaluation":
            return evaluate_anchor_checkpoint(spec, stage0_run_root=Path(stage0_run_root), output_root=output, device=device)
        result = run_behavioral_atlas_row(spec, device=device, output_root=output)
        _atomic_json(output / "rows" / "repair" / f"{spec['row_id']}.json", result)
        return result
    except Exception as exc:
        failure = {
            **spec,
            "status": "fail",
            "failure_category": "repair_execution_failure",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(output / "rows" / "repair" / f"{spec['row_id']}.json", failure)
        return failure


def _operational_pass(symmetry: dict[str, Any]) -> bool:
    if not symmetry.get("applicable", True):
        return True
    return (
        float(symmetry.get("operational_top1_flip_rate", math.inf)) <= BF16_TOP1_TOLERANCE
        and float(symmetry.get("operational_predictive_kl", math.inf)) <= BF16_PREDICTIVE_KL_TOLERANCE
    )


def merged_repair_audit(
    *,
    stage0_run_root: str | Path,
    stage0_manifest: str | Path,
    repair_root: str | Path,
    report_root: str | Path,
) -> dict[str, Any]:
    stage0_root = Path(stage0_run_root)
    repair_root = Path(repair_root)
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = [json.loads(line) for line in Path(stage0_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    original = load_results(stage0_root)
    repair_paths = sorted((repair_root / "rows" / "repair").glob("*.json"))
    repairs = [json.loads(path.read_text(encoding="utf-8")) for path in repair_paths]
    anchor = next((row for row in repairs if row.get("repair_kind") == "anchor_checkpoint_reevaluation"), None)
    compile_result = next((row for row in repairs if row.get("repair_kind") == "compile_candidate_no_cudagraph"), None)
    expected_noncompile = {str(row["row_id"]) for row in manifest_rows if row.get("profile_kind") != "compile_candidate"}
    observed = {str(row["row_id"]): row for row in original}
    original_audit = audit_results(original, manifest_rows)
    original_profiles = profile_metrics(original)
    scientific_and_required_profiles = [observed[row_id] for row_id in sorted(expected_noncompile) if row_id in observed]
    bf16_rows = scientific_and_required_profiles + ([compile_result] if compile_result and compile_result.get("status") == "pass" else [])
    standard = next((row for row in original if row.get("profile_kind") == "standard_trace"), None)
    compile_agreement = None
    compile_speedup = None
    compile_selected = False
    if compile_result and compile_result.get("status") == "pass" and standard:
        compile_agreement = {
            "validation_abs_difference": abs(float(compile_result["validation_loss"]) - float(standard["validation_loss"])),
            "test_abs_difference": abs(float(compile_result["test_loss"]) - float(standard["test_loss"])),
        }
        compile_speedup = float(compile_result["tokens_per_second"]) / max(float(standard["tokens_per_second"]), 1e-30)
        compile_selected = (
            bool(compile_result.get("execution", {}).get("compile_applied"))
            and max(compile_agreement.values()) <= 1e-3
            and compile_speedup >= 1.10
        )
    compile_resolved = compile_result is not None
    checks = {
        "original_noncompile_rows_complete": len(scientific_and_required_profiles) == len(expected_noncompile),
        "original_noncompile_rows_passed": bool(scientific_and_required_profiles) and all(row.get("status") == "pass" for row in scientific_and_required_profiles),
        "initial_state_identity": bool(original_audit["checks"].get("initial_states_identical_within_seed_and_identity_group")),
        "anchor_identity_within_seed_and_size": bool(original_audit["checks"].get("anchors_identical_within_seed_and_size")),
        "sample_order_identity": bool(original_audit["checks"].get("sample_order_identical_within_seed_and_budget")),
        "finite_metrics": bool(original_audit["checks"].get("finite_metrics")),
        "optimizer_provenance": bool(original_audit["checks"].get("executable_optimizer_provenance")),
        "trace_completeness": bool(original_audit["checks"].get("standard_trace_complete")),
        "freeze_integrity": bool(original_audit["checks"].get("freeze_integrity")),
        "semantic_permutation_identity": bool(original_audit["checks"].get("permutation_symmetry")),
        "bf16_prediction_behavior_gate": bool(bf16_rows) and all(_operational_pass(row.get("matched_key_expert_permutation", {})) for row in bf16_rows),
        "restart_identity": bool(original_audit["checks"].get("restart_identity_when_registered")),
        "repeatability": bool(original_profiles.get("repeatability_pass")),
        "trace_overhead_le_10_percent": bool(original_profiles.get("trace_overhead_pass")),
        "storage_forecast_with_25_percent_headroom_recorded": (stage0_root / "behavioral_atlas_forecast.json").is_file(),
        "anchor_sufficiency_fixed_bank_metrics": bool(anchor and anchor.get("status") == "pass" and anchor.get("anchor_sufficiency_pass")),
        "compile_path_decision_resolved": compile_resolved,
    }
    decision = "STAGE0_REPAIRED_PASS" if all(checks.values()) else "STAGE0_REPAIR_BLOCKED"
    summary = {
        "campaign": "phase6_behavioral_atlas_v2",
        "repair_version": REPAIR_VERSION,
        "decision": decision,
        "passed": decision == "STAGE0_REPAIRED_PASS",
        "checks": checks,
        "original_manifest": str(stage0_manifest),
        "original_manifest_sha256": hashlib.sha256(Path(stage0_manifest).read_bytes()).hexdigest(),
        "original_report_preserved": True,
        "original_observed_rows": len(original),
        "repair_rows_observed": len(repairs),
        "bf16_gate": {
            "version": "prediction_behavior_v2",
            "top1_flip_tolerance": BF16_TOP1_TOLERANCE,
            "predictive_kl_tolerance": BF16_PREDICTIVE_KL_TOLERANCE,
            "raw_absolute_logit_difference_role": "descriptive_only_scale_dependent",
        },
        "anchor_calibration": anchor,
        "compile_decision": {
            "selected_execution": "compiled_default_no_cudagraph" if compile_selected else "eager",
            "candidate_status": compile_result.get("status") if compile_result else "missing",
            "speedup_including_compile_cost": compile_speedup,
            "agreement": compile_agreement,
            "minimum_speedup": 1.10,
            "compile_is_optional_optimization": True,
        },
    }
    _atomic_json(repair_root / "stage0_repair_summary.json", summary)
    lines = [
        "# Phase 6.2 Stage 0 Measurement Repair",
        "",
        f"**Decision:** `{decision}`",
        "",
        "This revisioned audit preserves the original blocked Stage 0 evidence. It recalibrates only noninferential measurement gates and adds two registered repair evaluations.",
        "",
        "## Gate results",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in checks.items())
    lines.extend([
        "",
        "## BF16 operational invariance",
        "",
        "Strict FP32 matched key/expert permutation remains the semantic identity test. BF16 operational stability is gated by top-1 prediction changes and predictive KL; raw maximum logit difference is retained only as a scale-dependent diagnostic.",
        "",
        "## Anchor sufficiency",
        "",
        "The registered standard bank is 16,384 token states and the calibration bank is 32,768. Contribution, rank, entropy, and effective-support metrics must change by at most 5%. Dead-support fraction is reported at the fixed bank size but excluded from bank-doubling invariance because unseen occupancy changes mechanically with sample count.",
        "",
        "```json",
        json.dumps(anchor, indent=2, sort_keys=True),
        "```",
        "",
        "## Execution selection",
        "",
        f"Selected path: `{summary['compile_decision']['selected_execution']}`. Compilation is used only when stable, numerically consistent, and at least 10% faster; otherwise Stage 1 runs eagerly.",
        "",
        "## Interpretation boundary",
        "",
        "This repair is excluded from scientific inference. It validates measurement and execution only and does not compare model quality.",
    ])
    (report_root / "BEHAVIORAL_ATLAS_STAGE0_REPAIR_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_root / "BEHAVIORAL_ATLAS_STAGE0_REPAIR_LLM_HANDOFF.md").write_text(
        "# LLM handoff: Phase 6.2 Stage 0 repair\n\n"
        f"Decision: `{decision}`. Review `stage0_repair_summary.json` and the preserved original Stage 0 report. "
        "If passed, Stage 1 is the locked 168-row paired lifecycle study; do not use Stage 0 for scientific inference.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--stage0-manifest")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--index", type=int, required=True)
    run_parser.add_argument("--stage0-run-root", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--device", default="cuda")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--stage0-run-root", required=True)
    audit_parser.add_argument("--stage0-manifest", required=True)
    audit_parser.add_argument("--repair-root", required=True)
    audit_parser.add_argument("--report-root", required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        result = write_repair_manifest(args.output, args.stage0_manifest)
    elif args.command == "run":
        result = run_repair_manifest_row(
            args.manifest,
            args.index,
            stage0_run_root=args.stage0_run_root,
            output_root=args.output_root,
            device=args.device,
        )
    else:
        result = merged_repair_audit(
            stage0_run_root=args.stage0_run_root,
            stage0_manifest=args.stage0_manifest,
            repair_root=args.repair_root,
            report_root=args.report_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
    if args.command != "audit":
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
