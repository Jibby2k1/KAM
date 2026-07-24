from __future__ import annotations

from typing import Any

from torch import nn

from .baselines import GRULanguageModel, GRURegressor, MLPRegressor
from .model import KAMSequenceModel

LEGACY_MODELS = {"kam", "kernel-self", "memory-only", "dot-transformer", "dot-hybrid"}
PHASE2_BASE_VARIANTS = {"D0", "R0", "DD", "DR", "RR"}
PHASE2_VARIANTS = PHASE2_BASE_VARIANTS | {f"{base}-{suffix}" for base in {"DD", "DR", "RR"} for suffix in ("v", "a", "b")}
PHASE3_RANDOM_VARIANTS = {"RF-b", "RF-b-readout"}
PHASE3_STAGED_VARIANTS = {"DD-b-staged", "DR-b-staged"}
KAM_MODELS = LEGACY_MODELS | PHASE2_VARIANTS | PHASE3_RANDOM_VARIANTS | PHASE3_STAGED_VARIANTS


def _phase2_scores(label: str) -> tuple[str, str | None, bool, bool, str]:
    base, separator, suffix = label.partition("-")
    if base not in PHASE2_BASE_VARIANTS:
        raise ValueError(f"Unsupported Phase II variant: {label}")
    context, memory = {
        "D0": ("dot", None),
        "R0": ("radial", None),
        "DD": ("dot", "dot"),
        "DR": ("dot", "radial"),
        "RR": ("radial", "radial"),
    }[base]
    if separator:
        if base in {"D0", "R0"} or suffix not in {"v", "a", "b"}:
            raise ValueError(f"Unsupported Phase II memory-output suffix: {label}")
        memory_output = {"v": "residual", "a": "routes", "b": "both"}[suffix]
    else:
        memory_output = "both" if memory is not None else "residual"
    return context, memory, True, memory is not None, memory_output


def make_model(spec: dict[str, Any]) -> nn.Module:
    """Build a legacy or Phase II model from a JSON/YAML-serializable spec."""
    model_name = str(spec["model_name"])
    task_type = str(spec["task_type"])
    if model_name in PHASE3_STAGED_VARIANTS:
        # The schedule is enforced by the runner; this alias keeps the model
        # architecture and parameter count identical to its joint-training arm.
        staged_spec = dict(spec)
        staged_spec["model_name"] = model_name.removesuffix("-staged")
        return make_model(staged_spec)
    if model_name in PHASE3_RANDOM_VARIANTS:
        # Freeze the seeded support bank while retaining the DD-b architecture
        # and parameter budget. RF-b-readout freezes the rest of the backbone.
        random_spec = dict(spec)
        random_spec["model_name"] = "DD-b"
        model = make_model(random_spec)
        for block in getattr(model, "blocks", []):
            memory = getattr(block, "memory", None)
            if memory is not None:
                memory.memory_keys.requires_grad_(False)
                memory.memory_values.requires_grad_(False)
        if model_name == "RF-b-readout":
            for name, parameter in model.named_parameters():
                if not name.startswith("readout."):
                    parameter.requires_grad_(False)
        return model
    if model_name in KAM_MODELS:
        if model_name in PHASE2_VARIANTS:
            context_score, memory_score, use_context, use_memory, suffix_output = _phase2_scores(model_name)
            memory_output = suffix_output if "-" in model_name else str(spec.get("memory_output", spec.get("memory_mode", suffix_output)))
            if memory_output == "values":
                memory_output = "residual"
            expose_memory_weights = bool(spec.get("expose_memory_weights", False))
            score_type = context_score
        else:
            score_type = str(spec.get("score_type", "dot" if model_name.startswith("dot") else "rbf"))
            context_score = spec.get("context_score")
            memory_score = spec.get("memory_score")
            use_context = model_name not in {"memory-only"}
            use_memory = model_name in {"kam", "memory-only", "dot-hybrid"}
            memory_output = str(spec.get("memory_output", "residual"))
            expose_memory_weights = bool(spec.get("expose_memory_weights", True))

        pool = "mean" if model_name == "memory-only" and task_type == "regression" else spec.get("regression_pool", "last")
        return KAMSequenceModel(
            task=task_type,
            d_model=int(spec["d_model"]),
            num_heads=int(spec["num_heads"]),
            num_layers=int(spec["num_layers"]),
            num_supports=int(spec.get("num_supports", 64)),
            score_type=score_type,
            context_score=context_score,
            memory_score=memory_score,
            context_normalize_qk=spec.get("context_normalize_qk"),
            memory_normalize_qk=spec.get("memory_normalize_qk"),
            radial_metric=str(spec.get("radial_metric", "diagonal")),
            bandwidth=str(spec.get("bandwidth", "learned")),
            init_bandwidth=float(spec.get("bandwidth_init", spec.get("init_bandwidth", 1.0))),
            context_window=spec.get("context_window"),
            dropout=float(spec.get("dropout", 0.0)),
            max_seq_len=int(spec["max_seq_len"]),
            vocab_size=spec.get("vocab_size"),
            input_dim=spec.get("input_dim"),
            output_dim=int(spec.get("output_dim", 1)),
            use_context=use_context,
            use_memory=use_memory,
            regression_pool=pool,
            position_mode=str(spec.get("position_mode", "learned")),
            expose_memory_weights=expose_memory_weights,
            memory_output=memory_output,
            route_features=str(spec.get("route_features", "raw")),
            route_projection_dim=spec.get("route_projection_dim"),
            ffn_expansion=int(spec.get("ffn_expansion", 4)),
            parameter_match_target=spec.get("parameter_match_target"),
        )

    if model_name == "gru":
        if task_type == "language":
            return GRULanguageModel(vocab_size=int(spec["vocab_size"]), d_model=int(spec["d_model"]), num_layers=int(spec.get("num_layers", 1)))
        return GRURegressor(input_dim=int(spec["input_dim"]), d_model=int(spec["d_model"]), num_layers=int(spec.get("num_layers", 1)))

    if model_name == "mlp":
        if task_type != "regression":
            raise ValueError("The MLP baseline is implemented only for regression.")
        return MLPRegressor(window=int(spec["max_seq_len"]), input_dim=int(spec["input_dim"]), hidden=int(spec.get("mlp_hidden", 128)))

    raise ValueError(f"Unsupported model_name: {model_name}")
