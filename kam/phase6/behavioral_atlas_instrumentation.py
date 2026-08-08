"""Stage 0 anchor, functional, routing, optimizer, and symmetry diagnostics."""

from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kam.memory.routers import pairwise_scores
from kam.phase6.parameter_trace import GROUPS, grouped_named_parameters


@dataclass(frozen=True)
class AnchorBank:
    starts: Tensor
    inputs: Tensor
    targets: Tensor
    sha256: str
    token_states: int


@dataclass
class AnchorReference:
    queries: list[Tensor]
    updates: list[Tensor]
    logits: Tensor
    keys: list[Tensor]


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _strict_fp32_logits(model: nn.Module, sample: Tensor, device: torch.device) -> Tensor:
    """Evaluate semantic invariants without autocast or TF32 contraction."""
    if device.type != "cuda":
        return model(sample).float()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return model(sample).float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def build_anchor_bank(
    tokens: Tensor,
    validation_range: tuple[int, int],
    *,
    sequence_length: int,
    token_states: int,
    seed: int,
) -> AnchorBank:
    """Build and hash immutable validation windows used at every checkpoint."""
    if token_states <= 0:
        empty = torch.empty((0, sequence_length), dtype=torch.long)
        return AnchorBank(torch.empty(0, dtype=torch.long), empty, empty, hashlib.sha256(b"empty-anchor").hexdigest(), 0)
    low, high = (int(value) for value in validation_range)
    if high - low <= sequence_length + 1:
        raise ValueError("validation range is too short for the anchor bank")
    sequences = math.ceil(token_states / sequence_length)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    starts = torch.randint(low, high - sequence_length - 1, (sequences,), generator=generator)
    offsets = torch.arange(sequence_length)
    positions = starts[:, None] + offsets[None, :]
    inputs = tokens[positions].to(device="cpu", dtype=torch.long).contiguous()
    targets = tokens[positions + 1].to(device="cpu", dtype=torch.long).contiguous()
    digest = hashlib.sha256()
    for name, value in (("starts", starts), ("inputs", inputs), ("targets", targets)):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return AnchorBank(starts, inputs, targets, digest.hexdigest(), int(inputs.numel()))


def _capture_anchor_tensors(
    model: nn.Module,
    bank: AnchorBank,
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> tuple[float | None, list[Tensor], list[Tensor], Tensor]:
    layers = list(getattr(model, "memory_layers", []))
    query_parts: list[list[Tensor]] = [[] for _ in layers]
    update_parts: list[list[Tensor]] = [[] for _ in layers]
    logit_parts: list[Tensor] = []
    batch_queries: dict[int, Tensor] = {}
    batch_updates: dict[int, Tensor] = {}
    hooks = []
    for index, layer in enumerate(layers):
        def capture_query(_module, inputs, layer_index=index):
            batch_queries[layer_index] = inputs[0].detach()

        def capture_update(_module, _inputs, output, layer_index=index):
            value = output[0] if isinstance(output, tuple) else output
            batch_updates[layer_index] = value.detach()

        hooks.append(layer.register_forward_pre_hook(capture_query))
        hooks.append(layer.register_forward_hook(capture_update))
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for start in range(0, bank.inputs.shape[0], max(int(batch_size), 1)):
            stop = min(start + max(int(batch_size), 1), bank.inputs.shape[0])
            inputs = bank.inputs[start:stop].to(device, non_blocking=True)
            targets = bank.targets[start:stop].to(device, non_blocking=True)
            batch_queries.clear(); batch_updates.clear()
            with _autocast(device, precision):
                logits = model(inputs)
            loss_sum += float(F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction="sum"))
            token_count += targets.numel()
            logit_parts.append(logits.detach().to(device="cpu", dtype=torch.float16))
            for index in range(len(layers)):
                query_parts[index].append(batch_queries[index].reshape(-1, batch_queries[index].shape[-1]).to(device="cpu", dtype=torch.float16))
                update_parts[index].append(batch_updates[index].reshape(-1, batch_updates[index].shape[-1]).to(device="cpu", dtype=torch.float16))
    for hook in hooks:
        hook.remove()
    if was_training:
        model.train()
    queries = [torch.cat(parts) if parts else torch.empty(0) for parts in query_parts]
    updates = [torch.cat(parts) if parts else torch.empty(0) for parts in update_parts]
    logits = torch.cat(logit_parts) if logit_parts else torch.empty(0)
    return (loss_sum / token_count if token_count else None), queries, updates, logits


def _linear_cka(initial: Tensor, current: Tensor) -> float | None:
    if initial.numel() == 0 or current.numel() == 0 or initial.shape != current.shape:
        return None
    x = initial.float(); y = current.float()
    x = x - x.mean(0, keepdim=True); y = y - y.mean(0, keepdim=True)
    cross = x.T @ y
    xx = x.T @ x; yy = y.T @ y
    denominator = float(xx.square().sum().sqrt() * yy.square().sum().sqrt())
    if denominator <= 1e-30:
        return None
    return float(cross.square().sum()) / denominator


def _rank_metrics(values: Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"stable_rank": 0.0, "participation_ratio": 0.0, "normalized_spectral_entropy": 0.0}
    centered = values.float() - values.float().mean(0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0], 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if float(total) <= 1e-30:
        return {"stable_rank": 0.0, "participation_ratio": 0.0, "normalized_spectral_entropy": 0.0}
    probabilities = eigenvalues / total
    nonzero = probabilities > 1e-12
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    return {
        "stable_rank": float(total / eigenvalues.max().clamp_min(1e-30)),
        "participation_ratio": float(total.square() / eigenvalues.square().sum().clamp_min(1e-30)),
        "normalized_spectral_entropy": float(entropy / max(math.log(max(int(nonzero.sum()), 2)), 1e-30)),
    }


def _gini(values: Tensor) -> float:
    vector = values.detach().float().flatten().sort().values
    total = vector.sum()
    if vector.numel() == 0 or float(total) <= 0:
        return 0.0
    indices = torch.arange(1, vector.numel() + 1, device=vector.device, dtype=vector.dtype)
    return float((2 * (indices * vector).sum()) / (vector.numel() * total) - (vector.numel() + 1) / vector.numel())


def _route(query: Tensor, keys: Tensor, *, top_k: int, metric: str, temperature: float) -> tuple[Tensor, Tensor, Tensor]:
    scores = pairwise_scores(query, keys, metric=metric)  # type: ignore[arg-type]
    values, indices = scores.topk(min(int(top_k), keys.shape[0]), dim=-1)
    weights = torch.softmax(values / max(float(temperature), 1e-8), dim=-1)
    return indices, weights, values


def _route_overlap(reference_indices: Tensor, reference_weights: Tensor, indices: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    matches = indices.unsqueeze(-1) == reference_indices.unsqueeze(-2)
    intersections = matches.any(-1).sum(-1)
    unions = indices.shape[-1] + reference_indices.shape[-1] - intersections
    jaccard = intersections.float() / unions.clamp_min(1).float()
    weighted = (torch.minimum(weights.unsqueeze(-1), reference_weights.unsqueeze(-2)) * matches).sum(dim=(-1, -2))
    return jaccard, weighted


def _routing_state_metrics(
    query: Tensor,
    keys: Tensor,
    reference_indices: Tensor,
    reference_weights: Tensor,
    *,
    top_k: int,
    metric: str,
    temperature: float,
) -> tuple[dict[str, float], Tensor]:
    indices, weights, scores = _route(query, keys, top_k=top_k, metric=metric, temperature=temperature)
    jaccard, weighted = _route_overlap(reference_indices, reference_weights, indices, weights)
    counts = torch.bincount(indices.flatten(), minlength=keys.shape[0]).float()
    route_entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
    margin = scores[:, 0] - scores[:, 1] if scores.shape[-1] > 1 else None
    return {
        "jaccard_to_Q0_K0": float(jaccard.mean()),
        "weighted_overlap_to_Q0_K0": float(weighted.mean()),
        "mean_route_entropy": float(route_entropy.mean()),
        "mean_top1_top2_margin": float(margin.mean()) if margin is not None else 0.0,
    }, counts


def routing_decomposition(
    initial_queries: list[Tensor],
    current_queries: list[Tensor],
    initial_keys: list[Tensor],
    current_keys: list[Tensor],
    *,
    device: torch.device,
    top_k: int,
    metric: str,
    temperature: float,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    states = ("Q0_K0", "Qt_K0", "Q0_Kt", "Qt_Kt")
    per_layer: list[dict[str, Any]] = []
    for q0_cpu, qt_cpu, k0_cpu, kt_cpu in zip(initial_queries, current_queries, initial_keys, current_keys):
        k0 = k0_cpu.to(device=device, dtype=torch.float32)
        kt = kt_cpu.to(device=device, dtype=torch.float32)
        accum: dict[str, list[tuple[int, dict[str, float]]]] = {state: [] for state in states}
        support_counts = {state: torch.zeros(k0.shape[0], device=device) for state in states}
        for start in range(0, q0_cpu.shape[0], chunk_size):
            q0 = q0_cpu[start : start + chunk_size].to(device=device, dtype=torch.float32)
            qt = qt_cpu[start : start + chunk_size].to(device=device, dtype=torch.float32)
            reference_indices, reference_weights, _ = _route(q0, k0, top_k=top_k, metric=metric, temperature=temperature)
            for name, query, keys in (("Q0_K0", q0, k0), ("Qt_K0", qt, k0), ("Q0_Kt", q0, kt), ("Qt_Kt", qt, kt)):
                metrics, counts = _routing_state_metrics(query, keys, reference_indices, reference_weights, top_k=top_k, metric=metric, temperature=temperature)
                accum[name].append((query.shape[0], metrics))
                support_counts[name] += counts
        layer: dict[str, Any] = {}
        for state, chunks in accum.items():
            if not chunks:
                layer[state] = {}
                continue
            observations = sum(count for count, _ in chunks)
            layer[state] = {
                key: float(sum(count * metrics[key] for count, metrics in chunks) / max(observations, 1))
                for key in chunks[0][1]
            }
            counts = support_counts[state]
            probabilities = counts / counts.sum().clamp_min(1)
            entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()
            layer[state].update({
                "global_effective_support_count": float(1 / probabilities.square().sum().clamp_min(1e-30)),
                "normalized_support_entropy": float(entropy / max(math.log(counts.numel()), 1e-30)),
                "support_frequency_gini": _gini(counts),
                "dead_support_fraction": float((counts == 0).float().mean()),
            })
        query_churn = 1 - layer["Qt_K0"].get("jaccard_to_Q0_K0", 1.0)
        key_churn = 1 - layer["Q0_Kt"].get("jaccard_to_Q0_K0", 1.0)
        realized_churn = 1 - layer["Qt_Kt"].get("jaccard_to_Q0_K0", 1.0)
        layer["query_key_interaction_churn"] = float(realized_churn - query_churn - key_churn)
        per_layer.append(layer)
    means: dict[str, dict[str, float]] = {}
    for state in states:
        keys = per_layer[0][state].keys() if per_layer else ()
        means[state] = {key: float(np.mean([layer[state][key] for layer in per_layer])) for key in keys}
    return {
        "states": means,
        "query_key_interaction_churn": float(np.mean([layer["query_key_interaction_churn"] for layer in per_layer])) if per_layer else 0.0,
        "per_layer": per_layer,
    }


def evaluate_anchor_behavior(
    model: nn.Module,
    bank: AnchorBank,
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
    reference: AnchorReference | None = None,
    router_metric: str = "dot",
    router_temperature: float = 1.0,
    top_k: int = 4,
) -> tuple[dict[str, Any], AnchorReference]:
    loss, queries, updates, logits = _capture_anchor_tensors(
        model, bank, batch_size=batch_size, device=device, precision=precision
    )
    keys = [layer.keys.detach().to(device="cpu", dtype=torch.float32).clone() for layer in getattr(model, "memory_layers", [])]
    if reference is None:
        reference = AnchorReference(
            queries=[value.clone() for value in queries],
            updates=[value.clone() for value in updates],
            logits=logits.clone(),
            keys=[value.clone() for value in keys],
        )
    contribution = [float(update.float().norm() / query.float().norm().clamp_min(1e-30)) for query, update in zip(queries, updates)]
    query_cka = [_linear_cka(initial, current) for initial, current in zip(reference.queries, queries)]
    update_cka = [_linear_cka(initial, current) for initial, current in zip(reference.updates, updates)]
    ranks = [_rank_metrics(update) for update in updates]
    if logits.numel() and reference.logits.shape == logits.shape:
        initial_logits = reference.logits.float(); current_logits = logits.float()
        initial_probability = initial_logits.softmax(-1)
        logit_l2 = float((current_logits - initial_logits).square().sum(-1).sqrt().mean())
        predictive_kl = float((initial_probability * (initial_logits.log_softmax(-1) - current_logits.log_softmax(-1))).sum(-1).mean())
        top1_flip = float((initial_logits.argmax(-1) != current_logits.argmax(-1)).float().mean())
    else:
        logit_l2 = predictive_kl = top1_flip = 0.0
    routing = routing_decomposition(
        reference.queries,
        queries,
        reference.keys,
        keys,
        device=device,
        top_k=top_k,
        metric=router_metric,
        temperature=router_temperature,
    ) if keys else {"states": {}, "query_key_interaction_churn": 0.0, "per_layer": []}
    metrics = {
        "anchor_schema_version": 1,
        "anchor_sha256": bank.sha256,
        "anchor_token_states": bank.token_states,
        "anchor_loss": loss,
        "memory_contribution_ratio_mean": float(np.mean(contribution)) if contribution else 0.0,
        "memory_contribution_ratio_by_layer": contribution,
        "query_cka_to_initial_mean": float(np.mean([value for value in query_cka if value is not None])) if any(value is not None for value in query_cka) else None,
        "query_cka_to_initial_by_layer": query_cka,
        "memory_output_cka_to_initial_mean": float(np.mean([value for value in update_cka if value is not None])) if any(value is not None for value in update_cka) else None,
        "memory_output_cka_to_initial_by_layer": update_cka,
        "memory_output_rank_by_layer": ranks,
        "memory_output_stable_rank_mean": float(np.mean([value["stable_rank"] for value in ranks])) if ranks else 0.0,
        "memory_output_participation_ratio_mean": float(np.mean([value["participation_ratio"] for value in ranks])) if ranks else 0.0,
        "memory_output_spectral_entropy_mean": float(np.mean([value["normalized_spectral_entropy"] for value in ranks])) if ranks else 0.0,
        "anchor_logit_l2_drift": logit_l2,
        "anchor_predictive_kl_to_initial": predictive_kl,
        "anchor_top1_flip_rate": top1_flip,
        "routing_decomposition": routing,
    }
    return metrics, reference


def matched_key_expert_permutation_check(
    model: nn.Module,
    inputs: Tensor,
    *,
    device: torch.device,
    precision: str,
    seed: int,
) -> dict[str, Any]:
    layers = list(getattr(model, "memory_layers", []))
    if not layers:
        return {"applicable": False, "passed": True, "max_abs_logit_difference": 0.0, "mean_abs_logit_difference": 0.0}
    was_training = model.training; model.eval()
    sample = inputs[: min(inputs.shape[0], 4)].to(device)
    with torch.inference_mode():
        semantic_baseline = _strict_fp32_logits(model, sample, device)
        with _autocast(device, precision):
            operational_baseline = model(sample).float()
    snapshots: list[list[tuple[Tensor, Tensor]]] = []
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for layer in layers:
            support_count = int(layer.keys.shape[0])
            permutation = torch.randperm(support_count, generator=generator, device="cpu").to(layer.keys.device)
            layer_snapshots: list[tuple[Tensor, Tensor]] = []
            tensors: list[Tensor] = [layer.keys]
            tensors.extend(parameter for parameter in layer.experts.parameters() if parameter.ndim > 0 and parameter.shape[0] == support_count)
            for tensor in tensors:
                original = tensor.detach().clone()
                layer_snapshots.append((tensor, original))
                tensor.copy_(original[permutation])
            snapshots.append(layer_snapshots)
    try:
        with torch.inference_mode():
            semantic_permuted = _strict_fp32_logits(model, sample, device)
            with _autocast(device, precision):
                operational_permuted = model(sample).float()
    finally:
        with torch.no_grad():
            for layer_snapshots in snapshots:
                for tensor, original in layer_snapshots:
                    tensor.copy_(original)
        if was_training:
            model.train()
    semantic_difference = (semantic_baseline - semantic_permuted).abs()
    operational_difference = (operational_baseline - operational_permuted).abs()
    semantic_tolerance = 2e-5
    operational_top1_tolerance = 2e-2 if precision in {"bf16", "fp16"} else 0.0
    operational_kl_tolerance = 1e-3 if precision in {"bf16", "fp16"} else 1e-8
    semantic_maximum = float(semantic_difference.max())
    operational_maximum = float(operational_difference.max())
    baseline_probability = operational_baseline.softmax(-1)
    operational_predictive_kl = float(
        (
            baseline_probability
            * (operational_baseline.log_softmax(-1) - operational_permuted.log_softmax(-1))
        )
        .sum(-1)
        .mean()
    )
    operational_top1_flip = float(
        (operational_baseline.argmax(-1) != operational_permuted.argmax(-1)).float().mean()
    )
    return {
        "applicable": True,
        "passed": semantic_maximum <= semantic_tolerance,
        "semantic_precision": "fp32",
        "tolerance": semantic_tolerance,
        "max_abs_logit_difference": semantic_maximum,
        "mean_abs_logit_difference": float(semantic_difference.mean()),
        "operational_precision": precision,
        # Absolute logit differences are retained as a scale-dependent
        # diagnostic. The operational gate uses decision behavior and
        # distributional divergence; strict FP32 tests semantic identity.
        "operational_gate_version": "prediction_behavior_v2",
        "operational_top1_flip_tolerance": operational_top1_tolerance,
        "operational_predictive_kl_tolerance": operational_kl_tolerance,
        "operational_within_expected_precision_tolerance": (
            operational_top1_flip <= operational_top1_tolerance
            and operational_predictive_kl <= operational_kl_tolerance
        ),
        "operational_max_abs_logit_difference": operational_maximum,
        "operational_mean_abs_logit_difference": float(operational_difference.mean()),
        "operational_top1_flip_rate": operational_top1_flip,
        "operational_predictive_kl": operational_predictive_kl,
    }


def exact_step_update_metrics(model: nn.Module, pre_step: dict[str, Tensor]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for group, entries in grouped_named_parameters(model).items():
        update2 = weight2 = 0.0
        for name, parameter in entries:
            current = parameter.detach().float(); previous = pre_step[name].float()
            update2 += float((current - previous).square().sum())
            weight2 += float(current.square().sum())
        update = math.sqrt(update2); weight = math.sqrt(weight2)
        output[group] = {"optimizer_update_l2_norm": update, "update_to_weight_ratio": update / max(weight, 1e-30)}
    return output


class WindowDynamicsAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, list[float]]] = {group: {} for group in GROUPS}

    def observe(
        self,
        *,
        raw_gradients: dict[str, float],
        clipped_gradients: dict[str, float],
        updates: dict[str, dict[str, float]],
    ) -> None:
        for group in GROUPS:
            metrics = {
                "raw_gradient_l2_norm": float(raw_gradients.get(group, 0.0)),
                "clipped_gradient_l2_norm": float(clipped_gradients.get(group, 0.0)),
                **updates.get(group, {}),
            }
            for name, value in metrics.items():
                self.values[group].setdefault(name, []).append(float(value))

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=float)
        mean = float(array.mean()) if array.size else 0.0
        standard = float(array.std(ddof=1)) if array.size > 1 else 0.0
        return {
            "samples": int(array.size),
            "mean": mean,
            "median": float(np.median(array)) if array.size else 0.0,
            "p90": float(np.quantile(array, 0.90)) if array.size else 0.0,
            "variance": float(array.var(ddof=1)) if array.size > 1 else 0.0,
            "norm_snr": abs(mean) / standard if standard > 0 else None,
        }

    def summarize(self, *, reset: bool = True) -> dict[str, Any]:
        summary = {
            group: {name: self._summary(values) for name, values in metrics.items()}
            for group, metrics in self.values.items()
        }
        if reset:
            self.values = {group: {} for group in GROUPS}
        return summary


def optimizer_state_norms(model: nn.Module, optimizers: Iterable[torch.optim.Optimizer | None]) -> dict[str, float]:
    group_by_id = {id(parameter): group for group, entries in grouped_named_parameters(model).items() for _, parameter in entries}
    totals = {group: 0.0 for group in GROUPS}
    for optimizer in optimizers:
        if optimizer is None:
            continue
        for parameter, state in optimizer.state.items():
            group = group_by_id.get(id(parameter))
            if group is None:
                continue
            for value in state.values():
                if isinstance(value, Tensor) and value.shape == parameter.shape:
                    totals[group] += float(value.detach().float().square().sum())
    return {group: math.sqrt(total) for group, total in totals.items()}


__all__ = [
    "AnchorBank",
    "AnchorReference",
    "WindowDynamicsAccumulator",
    "build_anchor_bank",
    "evaluate_anchor_behavior",
    "exact_step_update_metrics",
    "matched_key_expert_permutation_check",
    "optimizer_state_norms",
    "routing_decomposition",
]
