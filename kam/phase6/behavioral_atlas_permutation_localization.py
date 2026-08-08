"""Independent replication and layer-interaction localization for Stage 1 permutation failures."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import socket
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from kam.phase6.behavioral_atlas_instrumentation import _strict_fp32_logits, build_anchor_bank
from kam.phase6.behavioral_atlas_permutation_audit import (
    BF16_KL_TOLERANCE,
    BF16_TOP1_TOLERANCE,
    SEMANTIC_TOLERANCE,
    _atomic_json,
    _difference_metrics,
    _operational_logits,
    _permutable_tensors,
)
from kam.phase6.behavioral_atlas_runner import build_behavioral_atlas_model
from kam.phase6.overnight_runner import _language_corpus

LOCALIZATION_VERSION = "stage1_permutation_localization_v1"
GPU_REPLICATES = 3


def _row_id(source_row_id: str, device_class: str, replicate: int) -> str:
    value = f"{LOCALIZATION_VERSION}:{source_row_id}:{device_class}:{replicate}"
    return "p6atlas_perm_localize_" + hashlib.sha256(value.encode()).hexdigest()[:16]


def write_manifest(audit_root: str | Path, output: str | Path) -> dict[str, Any]:
    summary_path = Path(audit_root) / "permutation_audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sources = sorted(str(value) for value in summary["reproduced_semantic_failures"])
    if len(sources) != 2:
        raise ValueError(f"localization requires exactly two reproduced strict failures, found {len(sources)}")
    specs = []
    for source in sources:
        for replicate in range(GPU_REPLICATES):
            specs.append({
                "row_id": _row_id(source, "gpu", replicate),
                "source_row_id": source,
                "device_class": "gpu",
                "replicate": replicate,
                "inferential": False,
                "retraining": False,
            })
    for source in sources:
        specs.append({
            "row_id": _row_id(source, "cpu", 0),
            "source_row_id": source,
            "device_class": "cpu",
            "replicate": 0,
            "inferential": False,
            "retraining": False,
        })
    payload = "".join(json.dumps(spec, sort_keys=True) + "\n" for spec in specs)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    return {
        "version": LOCALIZATION_VERSION,
        "rows": len(specs),
        "gpu_rows": sum(spec["device_class"] == "gpu" for spec in specs),
        "cpu_rows": sum(spec["device_class"] == "cpu" for spec in specs),
        "sources": sources,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def _fixed_permutations(model: nn.Module, seed: int) -> list[Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return [
        torch.randperm(int(layer.keys.shape[0]), generator=generator, device="cpu")
        for layer in getattr(model, "memory_layers", [])
    ]


def _subset_measurement(
    model: nn.Module,
    sample: Tensor,
    *,
    device: torch.device,
    precision: str,
    permutations: list[Tensor],
    layer_indices: tuple[int, ...],
    semantic_baseline: Tensor,
    operational_baseline: Tensor,
) -> dict[str, Any]:
    layers = list(getattr(model, "memory_layers", []))
    snapshots: list[tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for index in layer_indices:
            layer = layers[index]
            permutation = permutations[index].to(layer.keys.device)
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
        "layer_indices": list(layer_indices),
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


def _environment(device: torch.device) -> dict[str, Any]:
    payload = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
    }
    if device.type == "cuda":
        payload["gpu_name"] = torch.cuda.get_device_name(device)
        payload["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return payload


def localize_checkpoint(
    spec: dict[str, Any], *, stage1_root: Path, output_root: Path, device_name: str
) -> dict[str, Any]:
    source_path = stage1_root / "rows" / "behavioral_atlas_v2" / f"{spec['source_row_id']}.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    snapshot_path = Path(source["model_snapshots"][-1])
    if not snapshot_path.is_file():
        snapshot_path = stage1_root / "snapshots" / spec["source_row_id"] / "model_50000000.pt"
    checkpoint = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    row = checkpoint["row"]
    device = torch.device(device_name)
    model = build_behavioral_atlas_model(row).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokens, dataset = _language_corpus(row)
    bank = build_anchor_bank(
        tokens,
        tuple(int(value) for value in dataset["validation_range"]),
        sequence_length=int(row["sequence_length"]),
        token_states=int(row.get("anchor_token_states", 16_384)),
        seed=int(row["anchor_seed"]),
    )
    sample = bank.inputs[:4].to(device)
    precision = str(row.get("precision", "bf16")) if device.type == "cuda" else "fp32"
    with torch.inference_mode():
        semantic_baseline = _strict_fp32_logits(model, sample, device)
        operational_baseline = _operational_logits(model, sample, device, precision)
        baseline_repeats = [
            _difference_metrics(semantic_baseline, _strict_fp32_logits(model, sample, device))
            for _ in range(3)
        ]
    layer_count = len(getattr(model, "memory_layers", []))
    seed = int(row["seed"]) + 99
    permutations = _fixed_permutations(model, seed)
    full_indices = tuple(range(layer_count))
    full = _subset_measurement(
        model,
        sample,
        device=device,
        precision=precision,
        permutations=permutations,
        layer_indices=full_indices,
        semantic_baseline=semantic_baseline,
        operational_baseline=operational_baseline,
    )
    localization: dict[str, list[dict[str, Any]]] = {"single_layers": [], "layer_pairs": [], "prefixes": []}
    if device.type == "cuda":
        localization["single_layers"] = [
            _subset_measurement(
                model, sample, device=device, precision=precision, permutations=permutations,
                layer_indices=(index,), semantic_baseline=semantic_baseline,
                operational_baseline=operational_baseline,
            )
            for index in range(layer_count)
        ]
        localization["layer_pairs"] = [
            _subset_measurement(
                model, sample, device=device, precision=precision, permutations=permutations,
                layer_indices=indices, semantic_baseline=semantic_baseline,
                operational_baseline=operational_baseline,
            )
            for indices in itertools.combinations(range(layer_count), 2)
        ]
        localization["prefixes"] = [
            _subset_measurement(
                model, sample, device=device, precision=precision, permutations=permutations,
                layer_indices=tuple(range(stop)), semantic_baseline=semantic_baseline,
                operational_baseline=operational_baseline,
            )
            for stop in range(2, layer_count)
        ]
    result = {
        **spec,
        "version": LOCALIZATION_VERSION,
        "status": "pass",
        "arm": source["arm"],
        "seed": source["seed"],
        "permutation_seed": seed,
        "source_result_path": str(source_path),
        "source_snapshot_path": str(snapshot_path),
        "source_anchor_sha256": source.get("anchor_sha256"),
        "replication_anchor_sha256": bank.sha256,
        "sample_sequences": int(sample.shape[0]),
        "precision": precision,
        "environment": _environment(device),
        "baseline_repeatability": baseline_repeats,
        "full_matched_permutation": full,
        "localization": localization,
    }
    _atomic_json(output_root / "rows" / f"{spec['row_id']}.json", result)
    return result


def run_manifest_row(
    manifest: str | Path, index: int, *, stage1_root: str | Path, output_root: str | Path, device: str
) -> dict[str, Any]:
    specs = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    spec = specs[index]
    try:
        return localize_checkpoint(
            spec, stage1_root=Path(stage1_root), output_root=Path(output_root), device_name=device
        )
    except Exception as error:
        result = {**spec, "status": "fail", "exception_type": type(error).__name__, "exception": str(error)}
        _atomic_json(Path(output_root) / "rows" / f"{spec['row_id']}.json", result)
        raise


def aggregate(manifest: str | Path, output_root: str | Path, report_root: str | Path) -> dict[str, Any]:
    specs = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    rows = [
        json.loads(path.read_text())
        for path in sorted((Path(output_root) / "rows").glob("*.json"))
    ]
    expected = {spec["row_id"] for spec in specs}
    passed = [row for row in rows if row.get("status") == "pass"]
    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted({spec["source_row_id"] for spec in specs}):
        gpu = [row for row in passed if row["source_row_id"] == source and row["device_class"] == "gpu"]
        cpu = [row for row in passed if row["source_row_id"] == source and row["device_class"] == "cpu"]
        pair_failures = sorted({
            tuple(item["layer_indices"])
            for row in gpu
            for item in row["localization"]["layer_pairs"]
            if not item["semantic"]["passed"]
        })
        prefix_failures = sorted({
            tuple(item["layer_indices"])
            for row in gpu
            for item in row["localization"]["prefixes"]
            if not item["semantic"]["passed"]
        }, key=len)
        by_source[source] = {
            "gpu_replicates": len(gpu),
            "gpu_full_semantic_failures": sum(not row["full_matched_permutation"]["semantic"]["passed"] for row in gpu),
            "cpu_replicates": len(cpu),
            "cpu_full_semantic_failures": sum(not row["full_matched_permutation"]["semantic"]["passed"] for row in cpu),
            "single_layer_semantic_failures": sorted({
                item["layer_indices"][0]
                for row in gpu
                for item in row["localization"]["single_layers"]
                if not item["semantic"]["passed"]
            }),
            "pair_semantic_failures": [list(value) for value in pair_failures],
            "earliest_prefix_semantic_failure": list(prefix_failures[0]) if prefix_failures else None,
            "gpu_hosts": sorted({row["environment"]["hostname"] for row in gpu}),
            "gpu_full_max_abs_differences": [row["full_matched_permutation"]["semantic"]["max_abs_logit_difference"] for row in gpu],
            "cpu_full_max_abs_differences": [row["full_matched_permutation"]["semantic"]["max_abs_logit_difference"] for row in cpu],
        }
    complete = {row["row_id"] for row in passed} == expected
    gpu_all = complete and all(value["gpu_full_semantic_failures"] == GPU_REPLICATES for value in by_source.values())
    cpu_any = any(value["cpu_full_semantic_failures"] for value in by_source.values())
    if not complete:
        decision = "LOCALIZATION_EXECUTION_BLOCKED"
    elif not gpu_all:
        decision = "CUDA_STRICT_FAILURE_NOT_STABLE_ACROSS_PROCESSES"
    elif cpu_any:
        decision = "CROSS_DEVICE_STRICT_SEMANTIC_FAILURE"
    else:
        decision = "CUDA_MULTI_LAYER_NUMERICAL_ORDER_EFFECT"
    summary = {
        "version": LOCALIZATION_VERSION,
        "decision": decision,
        "all_rows_complete": complete,
        "expected_rows": len(specs),
        "observed_pass_rows": len(passed),
        "sources": by_source,
    }
    _atomic_json(Path(output_root) / "localization_summary.json", summary)
    report = Path(report_root)
    report.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 1 permutation failure localization",
        "",
        f"**Decision:** `{decision}`",
        "",
        f"Completed {len(passed)}/{len(specs)} independent checkpoint evaluations.",
        "",
    ]
    for source, value in by_source.items():
        lines.extend([
            f"## {source}",
            "",
            f"- GPU strict reproductions: {value['gpu_full_semantic_failures']}/{GPU_REPLICATES}",
            f"- CPU strict reproductions: {value['cpu_full_semantic_failures']}/{value['cpu_replicates']}",
            f"- Single-layer failures: {value['single_layer_semantic_failures']}",
            f"- Pair failures: {value['pair_semantic_failures']}",
            f"- Earliest failing cumulative prefix: {value['earliest_prefix_semantic_failure']}",
            f"- GPU hosts: {value['gpu_hosts']}",
            "",
        ])
    lines.append("This checkpoint-only localization does not revise the registered Stage 1 gate.")
    (report / "PERMUTATION_LOCALIZATION.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--audit-root", required=True)
    manifest.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--stage1-root", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--device", required=True)
    final = commands.add_parser("aggregate")
    final.add_argument("--manifest", required=True)
    final.add_argument("--output-root", required=True)
    final.add_argument("--report-root", required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        result = write_manifest(args.audit_root, args.output)
    elif args.command == "run":
        result = run_manifest_row(
            args.manifest, args.index, stage1_root=args.stage1_root,
            output_root=args.output_root, device=args.device,
        )
    else:
        result = aggregate(args.manifest, args.output_root, args.report_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
