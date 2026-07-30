"""Compact checkpoint-level parameter dynamics tracing without hot-path GPU synchronization."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn

GROUPS = ("memory_keys", "memory_experts", "memory_gates", "attention", "feedforward", "embeddings", "output_head")


def parameter_group(name: str) -> str:
    if "memory_layers" in name and name.endswith(".keys"):
        return "memory_keys"
    if "memory_layers" in name and ".experts." in name:
        return "memory_experts"
    if "memory_layers" in name and (".gate." in name or name.endswith(".global_value")):
        return "memory_gates"
    if ".attn." in name or "norm_attn" in name:
        return "attention"
    if ".ff." in name or "norm_ff" in name or "final_norm" in name:
        return "feedforward"
    if "token_embedding" in name or name.startswith("position."):
        return "embeddings"
    if name.startswith("lm_head."):
        return "output_head"
    return "feedforward"


def grouped_named_parameters(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {group: [] for group in GROUPS}
    for name, parameter in model.named_parameters():
        grouped[parameter_group(name)].append((name, parameter))
    return grouped


def tensor_hash(tensors: Iterable[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hash(model: nn.Module) -> str:
    return tensor_hash(sorted(model.state_dict().items()))


def key_hash(model: nn.Module) -> str:
    return tensor_hash((f"layer{index}.keys", layer.keys) for index, layer in enumerate(getattr(model, "memory_layers", [])))


def _clone_parameters(model: nn.Module) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _gradient_norms(grouped: dict[str, list[tuple[str, nn.Parameter]]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for group, entries in grouped.items():
        total = sum(float(parameter.grad.detach().float().square().sum()) for _, parameter in entries if parameter.grad is not None)
        output[group] = math.sqrt(total)
    return output


def _group_metrics(entries, initial, previous, pre_step, *, token_delta: int, changed_tolerance: float) -> dict[str, float]:
    norm2 = initial_norm2 = delta2 = incremental2 = update2 = dot = 0.0
    changed = elements = 0
    for name, parameter in entries:
        current, start, prior = parameter.detach().float(), initial[name].float(), previous[name].float()
        norm2 += float(current.square().sum()); initial_norm2 += float(start.square().sum())
        delta2 += float((current - start).square().sum()); incremental2 += float((current - prior).square().sum())
        dot += float((current * start).sum()); changed += int(((current - start).abs() > changed_tolerance).sum())
        elements += current.numel()
        if pre_step is not None:
            update2 += float((current - pre_step[name].float()).square().sum())
    norm, initial_norm = math.sqrt(norm2), math.sqrt(initial_norm2)
    cumulative, incremental = math.sqrt(delta2), math.sqrt(incremental2)
    return {
        "parameter_l2_norm": norm,
        "cumulative_l2_delta_from_initial": cumulative,
        "cumulative_relative_l2_delta_from_initial": cumulative / max(initial_norm, 1e-30),
        "cosine_similarity_to_initial": dot / max(norm * initial_norm, 1e-30),
        "incremental_l2_delta_from_previous_checkpoint": incremental,
        "incremental_relative_delta_per_million_tokens": incremental / max(initial_norm, 1e-30) / max(token_delta / 1_000_000.0, 1e-30),
        "update_to_weight_ratio": math.sqrt(update2) / max(norm, 1e-30),
        "fraction_elements_changed_above_tolerance": changed / max(elements, 1),
        "parameter_count": float(elements),
    }


def _memory_metrics(model: nn.Module, initial: dict[str, Tensor], probes: list[Tensor] | None) -> dict[str, Any]:
    angles: list[Tensor] = []
    ranks: list[float] = []; spectral: list[float] = []; neighbors: list[float] = []
    jaccards: list[float] = []; usage_entropy: list[float] = []; dead: list[float] = []; gates: list[float] = []
    layer_drifts: list[float] = []
    for index, layer in enumerate(getattr(model, "memory_layers", [])):
        current = layer.keys.detach().float(); start = initial[f"memory_layers.{index}.keys"].float()
        layer_drifts.append(float((current - start).norm()) / max(float(start.norm()), 1e-30))
        cosine = torch.nn.functional.cosine_similarity(current, start, dim=-1).clamp(-1.0, 1.0)
        angles.append(torch.acos(cosine).cpu())
        covariance = current.T @ current / max(current.shape[0], 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
        ranks.append(float((eigenvalues > eigenvalues.max() * 1e-6).sum()))
        spectral.append(float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()))
        sample_count = min(128, current.shape[0]); diagonal = torch.arange(sample_count, device=current.device)
        current_similarity = current[:sample_count] @ current.T; initial_similarity = start[:sample_count] @ start.T
        current_similarity[diagonal, diagonal] = -torch.inf; initial_similarity[diagonal, diagonal] = -torch.inf
        neighbors.append(float((current_similarity.argmax(1) == initial_similarity.argmax(1)).float().mean()))
        gate = getattr(getattr(layer, "gate", None), "scale", None)
        if isinstance(gate, Tensor): gates.append(float(gate.detach()))
        if probes is not None and index < len(probes):
            query = probes[index].to(current.device, dtype=current.dtype)
            k = min(int(getattr(layer.config, "top_k", 4)), current.shape[0])
            initial_topk = (query @ start.T).topk(k, dim=-1).indices
            current_topk = (query @ current.T).topk(k, dim=-1).indices
            intersections = (current_topk.unsqueeze(-1) == initial_topk.unsqueeze(-2)).any(-1).sum(-1)
            jaccards.append(float((intersections / (2 * k - intersections).clamp_min(1)).float().mean()))
            counts = torch.bincount(current_topk.flatten(), minlength=current.shape[0]).float(); used = counts > 0
            p = counts / counts.sum().clamp_min(1); entropy = -(p[used] * p[used].log()).sum()
            usage_entropy.append(float(entropy / max(math.log(current.shape[0]), 1e-30))); dead.append(float((~used).float().mean()))
    concatenated = torch.cat(angles) if angles else torch.zeros(1)
    return {
        "key_angular_displacement_median": float(concatenated.median()),
        "key_angular_displacement_q90": float(torch.quantile(concatenated, 0.90)),
        "key_effective_rank": float(np.mean(ranks)) if ranks else 0.0,
        "key_spectral_entropy": float(np.mean(spectral)) if spectral else 0.0,
        "nearest_neighbor_identity_retention": float(np.mean(neighbors)) if neighbors else 0.0,
        "routing_topk_jaccard_to_initial": float(np.mean(jaccards)) if jaccards else 0.0,
        "support_usage_entropy": float(np.mean(usage_entropy)) if usage_entropy else 0.0,
        "dead_support_fraction": float(np.mean(dead)) if dead else 0.0,
        "memory_gate_mean": float(np.mean(gates)) if gates else 0.0,
        "layer_relative_key_drift": layer_drifts,
    }


class ParameterTraceRecorder:
    def __init__(self, model: nn.Module, *, changed_tolerance: float = 1e-8) -> None:
        self.grouped = grouped_named_parameters(model); self.initial = _clone_parameters(model); self.previous = _clone_parameters(model)
        self.previous_tokens = 0; self.changed_tolerance = changed_tolerance

    def gradient_norms(self) -> dict[str, float]:
        return _gradient_norms(self.grouped)

    def pre_step_snapshot(self) -> dict[str, Tensor]:
        return {name: parameter.detach().clone() for entries in self.grouped.values() for name, parameter in entries}

    def record(self, model: nn.Module, *, arm: str, seed: int, tokens: int, step: int, phase: str,
               validation_loss: float, train_loss: float | None, probes: list[Tensor] | None,
               raw_gradients: dict[str, float] | None = None, clipped_gradients: dict[str, float] | None = None,
               pre_step: dict[str, Tensor] | None = None) -> dict[str, Any]:
        token_delta = tokens - self.previous_tokens
        groups = {group: {**_group_metrics(entries, self.initial, self.previous, pre_step, token_delta=token_delta,
                                             changed_tolerance=self.changed_tolerance),
                          "raw_gradient_l2_norm": float((raw_gradients or {}).get(group, 0.0)),
                          "clipped_gradient_l2_norm": float((clipped_gradients or {}).get(group, 0.0))}
                  for group, entries in self.grouped.items()}
        point = {"trace_schema_version": 1, "arm": arm, "seed": seed, "tokens": tokens, "step": step,
                 "phase": phase, "validation_loss": validation_loss, "train_loss": train_loss,
                 "key_hash": key_hash(model), "groups": groups, "memory": _memory_metrics(model, self.initial, probes)}
        self.previous = _clone_parameters(model); self.previous_tokens = tokens
        return point


__all__ = ["GROUPS", "ParameterTraceRecorder", "grouped_named_parameters", "key_hash", "parameter_group", "state_hash", "tensor_hash"]
