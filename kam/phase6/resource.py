from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceForecast:
    model_parameters: int
    memory_parameters: int
    active_parameters_per_token: int
    router_flops_per_token: int
    expert_flops_per_token: int
    estimated_parameter_bytes: int
    estimated_optimizer_bytes: int


def forecast_transformer_memory(
    *,
    d_model: int,
    n_layers: int,
    d_ff: int,
    num_supports: int,
    top_k: int,
    expert_mode: str = "vector",
    dtype_bytes: int = 4,
) -> ResourceForecast:
    model_parameters = n_layers * (4 * d_model * d_model + 2 * d_model * d_ff)
    if expert_mode == "vector":
        per_support = d_model
    elif expert_mode == "affine":
        per_support = d_model * d_model + d_model
    elif expert_mode == "low_rank":
        rank = max(1, d_model // 4)
        per_support = 2 * d_model * rank + d_model
    else:
        per_support = 0
    memory_parameters = num_supports * (d_model + per_support)
    active = top_k * per_support
    router_flops = 2 * num_supports * d_model
    expert_flops = top_k * max(per_support, d_model)
    return ResourceForecast(
        model_parameters=model_parameters,
        memory_parameters=memory_parameters,
        active_parameters_per_token=active,
        router_flops_per_token=router_flops,
        expert_flops_per_token=expert_flops,
        estimated_parameter_bytes=(model_parameters + memory_parameters) * dtype_bytes,
        estimated_optimizer_bytes=(model_parameters + memory_parameters) * dtype_bytes * 2,
    )
