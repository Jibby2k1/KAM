from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kam.memory import SparseMemoryConfig, SparseSeparableMemory
from kam.memory.routers import ChunkedExactTopKRouter, ExactTopKRouter, recall_at_k
from kam.optimization import ridge_solve, streaming_rls
from kam.phase6.diagnostics import finite_metrics, measure_forward, resource_accounting
from kam.transformer import ModernDecoder, TransformerConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _memory(row: dict[str, Any]) -> SparseSeparableMemory:
    router_chunk_size = row["router_chunk_size"] if row["router"] == "chunked" else None
    expert = {"vector": "vector", "affine": "affine", "low_rank": "low_rank"}[row["expert"]]
    config = SparseMemoryConfig(
        d_model=row["d_model"],
        num_supports=row["num_supports"],
        top_k=row["top_k"],
        expert_mode=expert,
        geometry_mode=row["geometry"],
        router_chunk_size=router_chunk_size,
    )
    return SparseSeparableMemory(config, seed=row["seed"])


def _route_check(row: dict[str, Any]) -> dict[str, float]:
    query = torch.randn(row["tokens"], row["d_model"])
    keys = torch.randn(row["num_supports"], row["d_model"])
    reference = ExactTopKRouter(row["top_k"])(query, keys)
    candidate = (
        ChunkedExactTopKRouter(row["top_k"], chunk_size=row["router_chunk_size"])(query, keys)
        if row["router"] == "chunked"
        else ExactTopKRouter(row["top_k"])(query, keys)
    )
    score = recall_at_k(candidate, reference)
    if score != 1.0 or not torch.equal(candidate.indices.sort(-1).values, reference.indices.sort(-1).values):
        raise AssertionError(f"exact/chunked route mismatch: recall={score}")
    return {"recall_at_k": score, "max_weight_error": float((candidate.weights - reference.weights).abs().max())}


def _gradient_check(row: dict[str, Any]) -> dict[str, float]:
    memory = _memory(row)
    query = torch.randn(2, row["tokens"], row["d_model"], requires_grad=True)
    target = torch.randn_like(query)
    output = memory(query)
    loss = (output - target).square().mean()
    loss.backward()
    gradients = [p.grad for p in memory.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(grad).all()) for grad in gradients):
        raise AssertionError("memory backward produced no finite gradients")
    return {"loss": float(loss.detach()), "gradient_tensors": float(len(gradients))}


def _zero_gate_check(row: dict[str, Any]) -> dict[str, float]:
    config = TransformerConfig(d_model=row["d_model"], n_heads=4, n_layers=1, vocab_size=23, max_seq_len=16)
    baseline = ModernDecoder(config)
    memory = _memory(row)
    with_memory = ModernDecoder(config, memory_layers=[memory])
    with_memory.load_state_dict(baseline.state_dict(), strict=False)
    tokens = torch.randint(0, config.vocab_size, (2, min(row["tokens"], 12)))
    baseline.eval()
    with_memory.eval()
    with torch.no_grad():
        expected = baseline(tokens)
        actual = with_memory(tokens)
    error = float((expected - actual).abs().max())
    if error > 1e-6:
        raise AssertionError(f"zero-gate baseline equivalence failed: {error}")
    return {"max_logit_error": error, "gate_scale": float(memory.gate.scale.detach())}


def _resource_check(row: dict[str, Any]) -> dict[str, float]:
    memory = _memory(row)
    metrics = resource_accounting(
        memory,
        tokens=row["tokens"],
        memory_slots=row["num_supports"],
        top_k=row["top_k"],
    )
    if metrics["total_parameters"] < metrics["active_parameters_per_token"]:
        raise AssertionError("active per-token count exceeds total memory parameter count")
    measured = measure_forward(
        memory,
        batch_size=1,
        sequence_length=row["tokens"],
        d_model=row["d_model"],
        repeats=3,
        warmup=1,
        device="auto",
    )
    metrics.update(measured)
    if metrics["forward_median_ms"] <= 0 or metrics["throughput_tokens_per_sec"] <= 0:
        raise AssertionError("forward timing measurement was non-positive")
    return metrics


def _ridge_check(row: dict[str, Any]) -> dict[str, float]:
    dtype = torch.float64
    features = torch.randn(48, 5, dtype=dtype)
    true = torch.randn(5, dtype=dtype)
    targets = features @ true + 0.01 * torch.randn(48, dtype=dtype)
    direct = ridge_solve(features, targets, regularization=1e-2, solver="cholesky")
    solve = ridge_solve(features, targets, regularization=1e-2, solver="solve")
    online = streaming_rls(features, targets, regularization=1e-2)
    direct_error = float((direct.solution - solve.solution).abs().max())
    online_error = float((direct.solution - online.solution).abs().max())
    if direct_error > 1e-8 or online_error > 1e-6:
        raise AssertionError(f"ridge solver mismatch: direct={direct_error}, online={online_error}")
    return {"direct_solve_error": direct_error, "streaming_error": online_error, "condition_number": direct.condition_number}


def _geometry_check(row: dict[str, Any]) -> dict[str, float]:
    memory = _memory(row)
    original = memory.keys.detach().clone()
    candidate = original + 10.0
    result = memory.update_geometry(candidate, trust_radius=0.1)
    if result["accepted"] or not torch.equal(original, memory.keys.detach()):
        raise AssertionError("trust-region rollback failed")
    nan_result = memory.update_geometry(torch.full_like(original, float("nan")))
    if nan_result["accepted"] or not torch.equal(original, memory.keys.detach()):
        raise AssertionError("non-finite geometry rollback failed")
    return {"trust_region_rejected": 1.0, "nonfinite_rejected": 1.0}


def _causal_check(row: dict[str, Any]) -> dict[str, float]:
    config = TransformerConfig(d_model=row["d_model"], n_heads=4, n_layers=1, vocab_size=23, max_seq_len=16)
    model = ModernDecoder(config).eval()
    prefix = torch.randint(0, config.vocab_size, (1, min(row["tokens"], 10)))
    altered = prefix.clone()
    altered[:, -1] = (altered[:, -1] + 1) % config.vocab_size
    with torch.no_grad():
        first = model(prefix)
        second = model(altered)
    if prefix.shape[1] > 1:
        error = float((first[:, :-1] - second[:, :-1]).abs().max())
    else:
        error = 0.0
    if error > 1e-6:
        raise AssertionError(f"causal mask leaked future token: {error}")
    return {"prefix_future_change_error": error}


CHECKS = {
    "route_exact_reference": _route_check,
    "route_chunked_reference": _route_check,
    "gradient_finite": _gradient_check,
    "zero_gate_equivalence": _zero_gate_check,
    "resource_accounting": _resource_check,
    "ridge_solver": _ridge_check,
    "geometry_rollback": _geometry_check,
    "causal_mask": _causal_check,
}


def run_row(row: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(row["seed"]))
    result = dict(row)
    result["status"] = "pass"
    try:
        metrics = CHECKS[row["task"]](row)
        result["metrics"] = metrics
        if not finite_metrics(metrics):
            raise AssertionError("non-finite Stage 0 metric")
    except Exception as exc:  # noqa: BLE001 - preserve a row-level failure for the gate report
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_manifest(manifest_path: str | Path, output_path: str | Path, row_index: int | None = None) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if row_index is not None:
        if row_index < 0 or row_index >= len(rows):
            raise IndexError(f"row index {row_index} outside manifest with {len(rows)} rows")
        rows = [rows[row_index]]
    results = [run_row(row) for row in rows]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    summary = {
        "manifest": str(manifest_path),
        "results": str(output),
        "rows": len(results),
        "passed": sum(result["status"] == "pass" for result in results),
        "failed": sum(result["status"] != "pass" for result in results),
    }
    summary_name = "run_summary.json" if row_index is None else f"{output.stem}_summary.json"
    output.with_name(summary_name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Phase 6 Stage 0 validity manifest")
    parser.add_argument("--manifest", type=Path, default=Path("results/phase6/stage0/manifests/validity.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/phase6/stage0/validity_results.jsonl"))
    parser.add_argument("--row-index", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_manifest(args.manifest, args.output, args.row_index), indent=2))


if __name__ == "__main__":
    main()
