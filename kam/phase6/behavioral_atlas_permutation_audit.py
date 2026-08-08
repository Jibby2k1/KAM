"""Checkpoint-only audit of Stage 1 permutation precision failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from kam.memory.routers import pairwise_scores
from kam.phase6.behavioral_atlas_instrumentation import (
    _strict_fp32_logits,
    build_anchor_bank,
)
from kam.phase6.behavioral_atlas_runner import build_behavioral_atlas_model
from kam.phase6.overnight_runner import _autocast, _language_corpus

AUDIT_VERSION = "stage1_permutation_checkpoint_audit_v1"
SEMANTIC_TOLERANCE = 2e-5
BF16_TOP1_TOLERANCE = 2e-2
BF16_KL_TOLERANCE = 1e-3


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _audit_id(source_row_id: str) -> str:
    digest = hashlib.sha256(f"{AUDIT_VERSION}:{source_row_id}".encode()).hexdigest()[:16]
    return f"p6atlas_perm_audit_{digest}"


def _operational_pass(row: dict[str, Any]) -> bool:
    check = row.get("matched_key_expert_permutation", {})
    return bool(check.get("operational_within_expected_precision_tolerance", True))


def _operational_boundary_score(row: dict[str, Any]) -> float:
    """Return the maximum registered BF16 criterion as a tolerance ratio."""
    check = row.get("matched_key_expert_permutation", {})
    top1 = float(check.get("operational_top1_flip_rate", 0.0)) / BF16_TOP1_TOLERANCE
    predictive_kl = float(check.get("operational_predictive_kl", 0.0)) / BF16_KL_TOLERANCE
    return max(top1, predictive_kl)


def build_audit_specs(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select strict failures and balanced BF16 boundary/worst examples per arm."""
    selected: dict[str, set[str]] = {}

    def include(row: dict[str, Any], role: str) -> None:
        selected.setdefault(str(row["row_id"]), set()).add(role)

    for row in results:
        if not row.get("matched_key_expert_permutation", {}).get("passed", True):
            include(row, "strict_fp32_failure")

    arms = sorted({str(row["arm"]) for row in results})
    for arm in arms:
        rows = [row for row in results if str(row["arm"]) == arm]
        failures = [row for row in rows if not _operational_pass(row)]
        passes = [row for row in rows if _operational_pass(row)]
        if failures:
            failures.sort(key=_operational_boundary_score)
            include(failures[0], "bf16_closest_failure")
            include(failures[-1], "bf16_worst_failure")
        if passes:
            passes.sort(key=_operational_boundary_score)
            include(passes[-1], "bf16_closest_pass")

    by_id = {str(row["row_id"]): row for row in results}
    specs = []
    for source_row_id, roles in sorted(selected.items()):
        source = by_id[source_row_id]
        check = source["matched_key_expert_permutation"]
        specs.append({
            "campaign": "phase6_behavioral_atlas_v2",
            "stage": AUDIT_VERSION,
            "row_id": _audit_id(source_row_id),
            "source_row_id": source_row_id,
            "arm": source["arm"],
            "seed": source["seed"],
            "selection_roles": sorted(roles),
            "original_semantic_pass": bool(check.get("passed", True)),
            "original_operational_pass": bool(check.get("operational_within_expected_precision_tolerance", True)),
            "original_semantic_max_abs_difference": float(check.get("max_abs_logit_difference", 0.0)),
            "original_operational_top1_flip_rate": float(check.get("operational_top1_flip_rate", 0.0)),
            "original_operational_predictive_kl": float(check.get("operational_predictive_kl", 0.0)),
            "inferential": False,
            "retraining": False,
        })
    return specs


def write_audit_manifest(stage1_root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(stage1_root) / "rows" / "behavioral_atlas_v2"
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]
    if len(results) != 168:
        raise ValueError(f"checkpoint audit requires 168 completed rows, found {len(results)}")
    specs = build_audit_specs(results)
    payload = "".join(json.dumps(spec, sort_keys=True) + "\n" for spec in specs)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    summary = {
        "audit_version": AUDIT_VERSION,
        "rows": len(specs),
        "arms": sorted({str(spec["arm"]) for spec in specs}),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "path": str(destination),
    }
    return summary


def _operational_logits(model: nn.Module, sample: Tensor, device: torch.device, precision: str) -> Tensor:
    with _autocast(device, precision):
        return model(sample).float()


def _difference_metrics(baseline: Tensor, candidate: Tensor) -> dict[str, float | bool]:
    difference = (baseline - candidate).abs()
    baseline_probability = baseline.softmax(-1)
    predictive_kl = float(
        (baseline_probability * (baseline.log_softmax(-1) - candidate.log_softmax(-1))).sum(-1).mean()
    )
    top1_flip = float((baseline.argmax(-1) != candidate.argmax(-1)).float().mean())
    return {
        "max_abs_logit_difference": float(difference.max()),
        "mean_abs_logit_difference": float(difference.mean()),
        "top1_flip_rate": top1_flip,
        "predictive_kl": predictive_kl,
    }


def _permutable_tensors(layer: nn.Module) -> list[Tensor]:
    support_count = int(layer.keys.shape[0])
    tensors: list[Tensor] = [layer.keys]
    tensors.extend(
        parameter
        for parameter in layer.experts.parameters()
        if parameter.ndim > 0 and parameter.shape[0] == support_count
    )
    return tensors


def _permutation_measurement(
    model: nn.Module,
    sample: Tensor,
    *,
    device: torch.device,
    precision: str,
    seed: int,
    layer_indices: list[int],
    semantic_baseline: Tensor,
    operational_baseline: Tensor,
) -> dict[str, Any]:
    layers = list(getattr(model, "memory_layers", []))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    snapshots: list[tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for index in layer_indices:
            layer = layers[index]
            permutation = torch.randperm(int(layer.keys.shape[0]), generator=generator, device="cpu").to(layer.keys.device)
            for tensor in _permutable_tensors(layer):
                original = tensor.detach().clone()
                snapshots.append((tensor, original))
                tensor.copy_(original[permutation])
    try:
        with torch.inference_mode():
            semantic_candidate = _strict_fp32_logits(model, sample, device)
            operational_candidate = _operational_logits(model, sample, device, precision)
    finally:
        with torch.no_grad():
            for tensor, original in snapshots:
                tensor.copy_(original)
    semantic = _difference_metrics(semantic_baseline, semantic_candidate)
    operational = _difference_metrics(operational_baseline, operational_candidate)
    return {
        "seed": int(seed),
        "layer_indices": layer_indices,
        "semantic": {
            **semantic,
            "tolerance": SEMANTIC_TOLERANCE,
            "passed": float(semantic["max_abs_logit_difference"]) <= SEMANTIC_TOLERANCE,
        },
        "operational": {
            **operational,
            "top1_tolerance": BF16_TOP1_TOLERANCE,
            "predictive_kl_tolerance": BF16_KL_TOLERANCE,
            "passed": (
                float(operational["top1_flip_rate"]) <= BF16_TOP1_TOLERANCE
                and float(operational["predictive_kl"]) <= BF16_KL_TOLERANCE
            ),
        },
    }


def _capture_layer_queries(model: nn.Module, sample: Tensor, device: torch.device) -> list[Tensor]:
    layers = list(getattr(model, "memory_layers", []))
    captured: list[Tensor | None] = [None] * len(layers)
    hooks = []
    for index, layer in enumerate(layers):
        def capture(_module, inputs, layer_index=index):
            captured[layer_index] = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).cpu()
        hooks.append(layer.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            _strict_fp32_logits(model, sample, device)
    finally:
        for hook in hooks:
            hook.remove()
    return [value if value is not None else torch.empty(0, layers[index].keys.shape[-1]) for index, value in enumerate(captured)]


def _route_margin_diagnostics(model: nn.Module, queries: list[Tensor]) -> list[dict[str, Any]]:
    output = []
    for index, (layer, query) in enumerate(zip(getattr(model, "memory_layers", []), queries)):
        scores = pairwise_scores(query, layer.keys.detach().float().cpu(), layer.config.metric)
        k = min(int(layer.config.top_k), scores.shape[-1])
        if k < scores.shape[-1]:
            values = torch.topk(scores, k=k + 1, dim=-1).values
            margins = (values[:, k - 1] - values[:, k]).float().numpy()
        else:
            margins = np.full((scores.shape[0],), np.inf)
        finite = margins[np.isfinite(margins)]
        output.append({
            "layer_index": index,
            "tokens": int(len(margins)),
            "margin_quantiles": {
                str(q): float(np.quantile(finite, q)) if finite.size else None
                for q in (0.0, 0.01, 0.05, 0.5, 0.95, 1.0)
            },
            "fraction_margin_le_1e_7": float(np.mean(margins <= 1e-7)),
            "fraction_margin_le_1e_6": float(np.mean(margins <= 1e-6)),
            "fraction_margin_le_1e_5": float(np.mean(margins <= 1e-5)),
        })
    return output


def audit_checkpoint(spec: dict[str, Any], *, stage1_root: Path, output_root: Path, device: str) -> dict[str, Any]:
    source_path = stage1_root / "rows" / "behavioral_atlas_v2" / f"{spec['source_row_id']}.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    snapshot_path = Path(source["model_snapshots"][-1])
    if not snapshot_path.is_file():
        snapshot_path = stage1_root / "snapshots" / str(spec["source_row_id"]) / "model_50000000.pt"
    payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    row = payload["row"]
    resolved_device = torch.device(device)
    model = build_behavioral_atlas_model(row).to(resolved_device)
    model.load_state_dict(payload["model"])
    model.eval()
    tokens, dataset = _language_corpus(row)
    validation_range = tuple(int(value) for value in dataset["validation_range"])
    bank = build_anchor_bank(
        tokens,
        validation_range,
        sequence_length=int(row["sequence_length"]),
        token_states=int(row.get("anchor_token_states", 16_384)),
        seed=int(row["anchor_seed"]),
    )
    sample = bank.inputs[:4].to(resolved_device)
    precision = str(row.get("precision", "bf16"))
    with torch.inference_mode():
        semantic_baseline = _strict_fp32_logits(model, sample, resolved_device)
        operational_baseline = _operational_logits(model, sample, resolved_device, precision)
        semantic_repeats = [
            _difference_metrics(semantic_baseline, _strict_fp32_logits(model, sample, resolved_device))
            for _ in range(3)
        ]
        operational_repeats = [
            _difference_metrics(operational_baseline, _operational_logits(model, sample, resolved_device, precision))
            for _ in range(3)
        ]
    layer_count = len(getattr(model, "memory_layers", []))
    base_seed = int(row["seed"]) + 99
    full = [
        _permutation_measurement(
            model,
            sample,
            device=resolved_device,
            precision=precision,
            seed=seed,
            layer_indices=list(range(layer_count)),
            semantic_baseline=semantic_baseline,
            operational_baseline=operational_baseline,
        )
        for seed in (base_seed, base_seed + 100, base_seed + 200)
    ]
    per_layer = [
        _permutation_measurement(
            model,
            sample,
            device=resolved_device,
            precision=precision,
            seed=base_seed,
            layer_indices=[index],
            semantic_baseline=semantic_baseline,
            operational_baseline=operational_baseline,
        )
        for index in range(layer_count)
    ]
    queries = _capture_layer_queries(model, sample, resolved_device)
    result = {
        **spec,
        "status": "pass",
        "source_result_path": str(source_path),
        "source_snapshot_path": str(snapshot_path),
        "source_anchor_sha256": source.get("anchor_sha256"),
        "audit_anchor_sha256": bank.sha256,
        "sample_sequences": int(sample.shape[0]),
        "sample_tokens": int(sample.numel()),
        "precision": precision,
        "baseline_repeatability": {
            "semantic_fp32": semantic_repeats,
            "operational": operational_repeats,
        },
        "full_matched_permutations": full,
        "per_layer_matched_permutations": per_layer,
        "route_margins": _route_margin_diagnostics(model, queries),
    }
    _atomic_json(output_root / "rows" / f"{spec['row_id']}.json", result)
    return result


def run_manifest_row(manifest: str | Path, index: int, *, stage1_root: str | Path, output_root: str | Path, device: str) -> dict[str, Any]:
    specs = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    spec = specs[index]
    try:
        return audit_checkpoint(spec, stage1_root=Path(stage1_root), output_root=Path(output_root), device=device)
    except Exception as error:
        result = {**spec, "status": "fail", "exception_type": type(error).__name__, "exception": str(error)}
        _atomic_json(Path(output_root) / "rows" / f"{spec['row_id']}.json", result)
        raise


def aggregate_audit(manifest: str | Path, output_root: str | Path, report_root: str | Path) -> dict[str, Any]:
    specs = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    output = Path(output_root)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "rows").glob("*.json"))]
    expected = {str(spec["row_id"]) for spec in specs}
    observed = {str(row["row_id"]) for row in rows}
    passed = [row for row in rows if row.get("status") == "pass"]
    baseline_semantic_max = max((float(metric["max_abs_logit_difference"]) for row in passed for metric in row["baseline_repeatability"]["semantic_fp32"]), default=math.inf)
    baseline_operational_kl_max = max((float(metric["predictive_kl"]) for row in passed for metric in row["baseline_repeatability"]["operational"]), default=math.inf)
    reproduced_semantic = [row for row in passed if not row["full_matched_permutations"][0]["semantic"]["passed"]]
    reproduced_operational = [row for row in passed if not row["full_matched_permutations"][0]["operational"]["passed"]]
    per_layer_semantic = [
        {"source_row_id": row["source_row_id"], "arm": row["arm"], "layer_index": item["layer_indices"][0], "metrics": item["semantic"]}
        for row in passed for item in row["per_layer_matched_permutations"] if not item["semantic"]["passed"]
    ]
    checks = {
        "all_manifest_rows_present": observed == expected,
        "all_audits_executed": len(passed) == len(specs),
        "semantic_baseline_repeatable": baseline_semantic_max <= SEMANTIC_TOLERANCE,
        "operational_baseline_repeatable": baseline_operational_kl_max <= 1e-8,
    }
    if not all(checks.values()):
        disposition = "AUDIT_EXECUTION_OR_REPEATABILITY_BLOCKED"
    elif reproduced_semantic:
        disposition = "STRICT_SEMANTIC_FAILURE_REPRODUCED"
    elif reproduced_operational:
        disposition = "BF16_PERMUTATION_ORDER_SENSITIVITY_REPRODUCED"
    else:
        disposition = "ORIGINAL_FAILURES_NOT_REPRODUCED"
    summary = {
        "audit_version": AUDIT_VERSION,
        "decision": disposition,
        "checks": checks,
        "expected_rows": len(specs),
        "observed_rows": len(rows),
        "missing_row_ids": sorted(expected - observed),
        "reproduced_semantic_failures": [row["source_row_id"] for row in reproduced_semantic],
        "reproduced_operational_failures": [row["source_row_id"] for row in reproduced_operational],
        "per_layer_semantic_failures": per_layer_semantic,
        "baseline_semantic_max_abs_difference": baseline_semantic_max,
        "baseline_operational_max_predictive_kl": baseline_operational_kl_max,
    }
    _atomic_json(output / "permutation_audit_summary.json", summary)
    report = Path(report_root)
    report.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6.2 Stage 1 permutation checkpoint audit",
        "",
        f"**Decision:** `{disposition}`",
        "",
        f"Audited {len(rows)}/{len(specs)} selected checkpoints without retraining.",
        "",
        "## Checks",
        "",
        *[f"- {name}: `{value}`" for name, value in checks.items()],
        "",
        "## Reproduction",
        "",
        f"- Strict semantic failures reproduced: {len(reproduced_semantic)}",
        f"- BF16 operational failures reproduced: {len(reproduced_operational)}",
        f"- Per-layer strict failures: {len(per_layer_semantic)}",
        f"- Baseline semantic repeatability maximum: {baseline_semantic_max:.6g}",
        f"- Baseline operational repeatability maximum KL: {baseline_operational_kl_max:.6g}",
        "",
        "This audit localizes measurement behavior only. It does not revise registered thresholds or reinterpret Stage 1 outcomes.",
    ]
    (report / "PERMUTATION_CHECKPOINT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--stage1-root", required=True)
    manifest_parser.add_argument("--output", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--index", type=int, required=True)
    run_parser.add_argument("--stage1-root", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--device", default="cuda")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--manifest", required=True)
    aggregate_parser.add_argument("--output-root", required=True)
    aggregate_parser.add_argument("--report-root", required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        result = write_audit_manifest(args.stage1_root, args.output)
    elif args.command == "run":
        result = run_manifest_row(args.manifest, args.index, stage1_root=args.stage1_root, output_root=args.output_root, device=args.device)
    else:
        result = aggregate_audit(args.manifest, args.output_root, args.report_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
