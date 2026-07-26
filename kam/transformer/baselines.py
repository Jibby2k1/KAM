from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from torch import nn

from kam.memory import SparseMemoryConfig, SparseSeparableMemory
from kam.memory.baselines import DualMemory, MemoryTokenLayer, MixtureOfExpertsMemory, ProductKeyMemory
from kam.memory.episodic import EpisodicMemory

from .config import TransformerConfig
from .decoder import ModernDecoder


SCALE_PARAMETER_TARGETS = {
    "2M": 2_000_000,
    "10M": 10_000_000,
    "30M": 30_000_000,
    "100M": 100_000_000,
}


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    family: str
    optimization: str = "joint_sgd"
    geometry: str = "none"
    value_mode: str = "none"


def scale_config(scale: str, *, vocab_size: int = 128, max_seq_len: int = 128) -> TransformerConfig:
    """Return a deterministic small-to-medium config for manifest controls.

    Exact production parameter targets are measured after construction; these
    values are intentionally conservative for development and HPG profiles.
    """
    table = {
        "tiny": (32, 4, 1),
        "2M": (128, 4, 3),
        "10M": (256, 8, 4),
        "30M": (384, 8, 6),
        "100M": (512, 8, 8),
    }
    if scale not in table:
        raise ValueError(f"unknown transformer scale: {scale}")
    d_model, n_heads, n_layers = table[scale]
    return TransformerConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, vocab_size=vocab_size, max_seq_len=max_seq_len)


def _decoder_parameter_count(config: TransformerConfig) -> int:
    """Count decoder parameters without constructing a candidate model."""
    d = config.d_model
    d_ff = config.feedforward_dim
    embedding_and_head = (2 * int(config.vocab_size or 0) + config.max_seq_len + 1) * d
    block = 4 * d * d + 3 * d * d_ff + 7 * d + 2 * d_ff
    return embedding_and_head + config.n_layers * block


def _memory_parameter_count(name: str, d_model: int, num_supports: int) -> int:
    """Return the parameter contribution of a named memory control."""
    supports = max(1, int(num_supports))
    if name == "T-MEMTOK":
        return supports * d_model + 1
    if name == "T-MOE":
        experts = min(supports, 4)
        hidden = 4 * d_model
        router = d_model * experts + experts
        expert = 2 * d_model * hidden + hidden + d_model
        return router + experts * expert
    if name == "T-PKM":
        codebook = max(2, int(supports**0.5))
        return codebook * d_model + codebook * codebook * d_model
    if name.startswith("T-KAM"):
        rank = max(4, d_model // 32)
        per_layer = supports * d_model + d_model + 1
        if name not in {"T-KAM-F"}:
            per_layer += supports * (2 * d_model * rank + d_model)
        else:
            per_layer += supports * d_model
        return per_layer
    return 0


def parameter_budget(scale: str) -> int:
    """Return the declared total-parameter budget for a transformer scale."""
    try:
        return SCALE_PARAMETER_TARGETS[scale]
    except KeyError as exc:
        raise ValueError(f"unknown transformer scale: {scale}") from exc


@lru_cache(maxsize=128)
def _matched_config(
    name: str,
    scale: str,
    vocab_size: int,
    max_seq_len: int,
    num_supports: int,
    target_parameters: int,
) -> TransformerConfig:
    """Choose a lightweight architecture config close to the scale budget.

    The search is analytic, so it does not instantiate hundreds of candidate
    models.  Different memory controls receive different backbone widths and
    FFN sizes, making the measured total-parameter field meaningful instead
    of silently comparing an enlarged model with a smaller baseline.
    """
    base = scale_config(scale, vocab_size=vocab_size, max_seq_len=max_seq_len)
    target = int(target_parameters)
    if target <= 0:
        raise ValueError("target_parameters must be positive")
    # Width only needs to be divisible by the attention head count. Searching
    # at 32-wide increments left small matched controls several percent off
    # budget, especially when memory parameters dominate.
    d_models = range(max(32, base.n_heads), min(1024, max(base.d_model * 2, 256)) + 1, base.n_heads)
    layers = range(1, min(12, max(base.n_layers * 2, 8)) + 1)
    # Keep the dense/control backbone recipe fixed at a standard 4x FFN.
    # T-WIDE is the only architecture allowed to spend its matched budget on
    # a wider FFN; otherwise analytic matching can silently make T0 equally
    # wide and erase the intended ordinary-capacity control.
    multipliers = (6, 8, 10, 12) if name == "T-WIDE" else (4,)
    candidates: list[tuple[int, int, int, int, int]] = []
    for d_model in d_models:
        for n_layers in layers:
            for multiplier in multipliers:
                d_ff = max(32, multiplier * d_model)
                config = TransformerConfig(
                    d_model=d_model,
                    n_heads=base.n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    vocab_size=vocab_size,
                    max_seq_len=max_seq_len,
                )
                count = _decoder_parameter_count(config) + n_layers * _memory_parameter_count(name, d_model, num_supports)
                candidates.append((abs(count - target), count, d_model, n_layers, d_ff))
    _, _, d_model, n_layers, d_ff = min(candidates, key=lambda item: (item[0], item[1]))
    return TransformerConfig(
        d_model=d_model,
        n_heads=base.n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )


def _memory_layers(name: str, config: TransformerConfig, *, num_supports: int = 32, top_k: int = 4, seed: int = 0) -> list[nn.Module]:
    if name == "T0" or name == "T-WIDE":
        return []
    if name == "T-MEMTOK":
        return [MemoryTokenLayer(config.d_model, num_tokens=num_supports, top_k=top_k) for _ in range(config.n_layers)]
    if name == "T-MOE":
        return [MixtureOfExpertsMemory(config.d_model, num_experts=min(num_supports, 4), top_k=min(top_k, 2)) for _ in range(config.n_layers)]
    if name == "T-PKM":
        codebook_size = max(2, int(num_supports**0.5))
        return [ProductKeyMemory(config.d_model, codebook_size=codebook_size, top_k=top_k) for _ in range(config.n_layers)]
    geometry = "fixed_random" if name == "T-KAM-F" else "learned_full"
    value_mode = "low_rank" if name in {"T-KAM-L", "T-KAM-ALT", "T-KAM-VP"} else "vector"
    sparse_layers: list[nn.Module] = []
    for index in range(config.n_layers):
        persistent = SparseSeparableMemory(
            SparseMemoryConfig(
                d_model=config.d_model,
                num_supports=num_supports,
                top_k=top_k,
                expert_mode=value_mode,
                expert_rank=max(4, config.d_model // 32),
                geometry_mode=geometry,
            ),
            seed=seed + index,
        )
        if name == "T-KAM-DUAL":
            sparse_layers.append(DualMemory(persistent, EpisodicMemory(capacity=num_supports, d_model=config.d_model)))
        else:
            sparse_layers.append(persistent)
    return sparse_layers


def architecture_spec(name: str) -> ArchitectureSpec:
    if name not in {"T0", "T-WIDE", "T-MEMTOK", "T-MOE", "T-PKM", "T-KAM-F", "T-KAM-L", "T-KAM-ALT", "T-KAM-VP", "T-KAM-ONLINE", "T-KAM-DUAL"}:
        raise ValueError(f"unknown Phase 6 architecture: {name}")
    if name.startswith("T-KAM"):
        optimization = {
            "T-KAM-ALT": "alternating_8_1",
            "T-KAM-VP": "variable_projection_stopgrad",
            "T-KAM-ONLINE": "online_adaptation",
            "T-KAM-DUAL": "joint_sgd",
        }.get(name, "joint_sgd")
        geometry = "fixed_random" if name == "T-KAM-F" else "learned_full"
        return ArchitectureSpec(name, "sparse_kam", optimization, geometry, "vector_or_low_rank")
    return ArchitectureSpec(name, name.lower().replace("-", "_"), value_mode="dense_or_baseline")


def build_baseline(
    name: str,
    *,
    scale: str = "tiny",
    vocab_size: int = 128,
    max_seq_len: int = 128,
    num_supports: int = 32,
    top_k: int = 4,
    seed: int = 0,
    target_parameters: int | None = None,
) -> tuple[ModernDecoder, ArchitectureSpec]:
    """Build every named Phase 6 architecture through one interface."""
    config = (
        _matched_config(name, scale, vocab_size, max_seq_len, num_supports, int(target_parameters))
        if target_parameters is not None
        else scale_config(scale, vocab_size=vocab_size, max_seq_len=max_seq_len)
    )
    if name == "T-WIDE" and target_parameters is None:
        config = TransformerConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=6 * config.d_model,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        )
    model = ModernDecoder(config, memory_layers=_memory_layers(name, config, num_supports=num_supports, top_k=top_k, seed=seed))
    return model, architecture_spec(name)


__all__ = ["ArchitectureSpec", "architecture_spec", "build_baseline", "parameter_budget", "scale_config"]
