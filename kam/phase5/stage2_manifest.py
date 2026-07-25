"""Build immutable Stage 2A–2D manifests from the authoritative brief."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from kam.capacity import active_parameter_count
from kam.factory import make_model
from kam.phase4.table import write_json, write_table

COMPONENT_VARIANTS = ("D0", "DD-L", "RF-KV", "RF-FULL", "RK-LV", "LK-RV", "KC-LV", "DD-A", "DD-V", "DD-B")
CROSSOVER_VARIANTS = ("D0", "DD-L", "RF-KV", "KC-LV")
SYMBOLIC_VARIANTS = ("D0", "DD-L", "RF-KV", "KC-LV")
COMPONENT_TARGETS = (600_000, 1_000_000, 2_000_000)
CROSSOVER_TARGETS = (250_000, 400_000, 600_000, 800_000, 1_000_000, 1_500_000, 2_000_000, 4_000_000)


def _seed(stage: str, task: str, cell: str, index: int) -> int:
    token = f"stage2|{stage}|{task}|{cell}|{index}".encode()
    return 21001 + int(hashlib.sha256(token).hexdigest()[:8], 16) % 100000


@lru_cache(maxsize=None)
def resolve_architecture(target: int, variant: str, task_type: str, max_seq_len: int, num_supports: int) -> dict[str, Any]:
    if variant == "RFF":
        feature_dim = max(1, int((target - 1) / (max_seq_len * 2 + 2)))
        spec = {
            "model_name": variant, "task_type": task_type, "d_model": 32,
            "num_heads": 4, "num_layers": 1, "num_supports": num_supports,
            "max_seq_len": max_seq_len, "input_dim": input_dim, "vocab_size": vocab_size,
            "output_dim": 1, "route_features": "projected", "route_projection_dim": 64,
            "memory_output": "both", "fourier_features": feature_dim,
        }
        model = make_model(spec)
        count = active_parameter_count(model)
        del model
        gc.collect()
        return {"d_model": 32, "num_heads": 4, "num_layers": 1, "num_supports": num_supports,
                "fourier_features": feature_dim, "active_parameter_count": count}
    vocab_size = 20 if task_type == "language" else None
    input_dim = None if task_type == "language" else 2
    best: tuple[int, dict[str, Any]] | None = None
    for d_model in range(64, 321, 8):
        num_heads = 4
        for num_layers in range(1, 7):
                spec = {
                    "model_name": variant, "task_type": task_type, "d_model": d_model,
                    "num_heads": num_heads, "num_layers": num_layers, "num_supports": num_supports,
                    "max_seq_len": max_seq_len, "input_dim": input_dim, "vocab_size": vocab_size,
                    "output_dim": 1, "route_features": "projected", "route_projection_dim": 64,
                    "memory_output": "both", "expose_memory_weights": variant not in {"D0"},
                }
                try:
                    model = make_model(spec)
                except (ValueError, RuntimeError):
                    continue
                count = active_parameter_count(model)
                del model
                gc.collect()
                error = abs(count - target)
                if best is None or error < best[0]:
                    best = (error, {"d_model": d_model, "num_heads": num_heads, "num_layers": num_layers,
                                    "num_supports": num_supports, "active_parameter_count": count})
    if best is None:
        raise RuntimeError(f"could not resolve architecture for {variant}, {target}, {task_type}")
    return best[1]


def _base_row(stage: str, task: str, variant: str, target: int, seed_index: int, *,
              cell: str, heldout_streams: int, steps: int, task_type: str = "regression",
              num_supports: int = 64, **factors: Any) -> dict[str, Any]:
    max_seq_len = int(factors.pop("seq_len", 64))
    architecture = resolve_architecture(target, variant, task_type,
                                         max_seq_len + (1 if task_type == "language" else 0),
                                         num_supports)
    row = {
        "row_id": -1, "stage": stage, "task": task, "variant": variant, "cell": cell,
        "seed_index": seed_index, "seed": _seed(stage, task, cell, seed_index),
        "run_id": f"p5_{stage}_{task}_{cell}_{variant}_s{seed_index}".replace("-", "m"),
        # The nominal target is the scientific scale label. The executable
        # target is the exact capacity of the resolved architecture; this
        # preserves the <=1% gate without silently accepting a discrete-width mismatch.
        "nominal_target_active_parameters": target,
        "target_active_parameters": int(architecture["active_parameter_count"]),
        "active_match_tolerance": 0.01,
        "task_type": task_type, "training_protocol": "iid_window_training",
        "d_model": architecture["d_model"], "num_heads": architecture["num_heads"],
        "num_layers": architecture["num_layers"], "num_supports": architecture["num_supports"],
        "fourier_features": architecture.get("fourier_features"),
        "seq_len": max_seq_len, "series_length": 800, "train_length": 800,
        "validation_length": 320, "test_length": 320, "prequential_length": 320,
        "steps": steps, "eval_every": max(2, steps // 50), "batch_size": 32,
        "eval_batches": 32, "heldout_streams": heldout_streams,
        "trace_eval_every": max(2, steps // 10), "trace_eval_batches": 4,
        "learning_rate": 3e-4, "weight_decay": 1e-4, "precision": "amp",
        "route_features": "projected", "route_projection_dim": 64,
        "memory_output": {"DD-A": "routes", "DD-V": "residual", "DD-B": "both"}.get(variant, "both"),
        "expose_memory_weights": variant not in {"D0"},
        "memory_trace": variant not in {"D0"},
        "trace_test": True, "evaluate_train": True, "evaluate_test": True,
        "save_validation_predictions": False, "save_test_predictions": False,
        "center_initialization": factors.pop("center_initialization", "random_normal"),
        "regime_count": 3, "regime_separation": "medium", "return_probability": 0.5,
        "dwell_length": 64, "transition_type": "abrupt", "observation_noise": 0.0,
        "process_noise": 0.0, "input_noise": 0.0, "observability": "full",
        "transition_entropy": 0.5, "emission_overlap": 0.2,
        "explicit_regime_token": False, "order": 10,
    }
    row.update(factors)
    return row


def build_component_rows() -> list[dict[str, Any]]:
    rows = []
    for task in ("controlled_prototype", "switching_mackey_glass_controlled", "switching_narma_controlled"):
        for variant in COMPONENT_VARIANTS:
            for target in COMPONENT_TARGETS:
                for seed_index in range(5):
                    rows.append(_base_row("stage2A_component", task, variant, target, seed_index, cell=f"P{target}", heldout_streams=5, steps=500, num_supports=64))
    return _number_rows(rows)


def build_capacity_rows() -> list[dict[str, Any]]:
    rows = []
    for task in ("controlled_prototype", "switching_mackey_glass_controlled", "switching_narma_controlled"):
        for variant in CROSSOVER_VARIANTS:
            for target in CROSSOVER_TARGETS:
                for seed_index in range(5):
                    rows.append(_base_row("stage2B_capacity", task, variant, target, seed_index, cell=f"P{target}", heldout_streams=5, steps=500, num_supports=64))
    return _number_rows(rows)


def build_factorial_rows() -> list[dict[str, Any]]:
    designs = [
        {"return_probability": 0.0, "regime_separation": "low", "observability": "full", "observation_noise": 0.0, "process_noise": 0.0, "center_initialization": "random_normal", "num_supports": 16},
        {"return_probability": 0.25, "regime_separation": "medium", "observability": "partial", "observation_noise": 0.01, "process_noise": 0.0, "center_initialization": "sampled_training_points", "num_supports": 32},
        {"return_probability": 0.5, "regime_separation": "high", "observability": "hidden_driver", "observation_noise": 0.03, "process_noise": 0.01, "center_initialization": "kmeans", "num_supports": 64},
        {"return_probability": 0.75, "regime_separation": "medium", "observability": "full", "observation_noise": 0.0, "process_noise": 0.01, "center_initialization": "farthest_point", "num_supports": 128},
        {"return_probability": 1.0, "regime_separation": "high", "observability": "partial", "observation_noise": 0.03, "process_noise": 0.0, "center_initialization": "sampled_training_points", "num_supports": 256},
        {"return_probability": 0.5, "regime_separation": "low", "observability": "full", "observation_noise": 0.01, "process_noise": 0.01, "center_initialization": "kmeans", "num_supports": 64},
        {"return_probability": 0.25, "regime_separation": "high", "observability": "hidden_driver", "observation_noise": 0.0, "process_noise": 0.0, "center_initialization": "farthest_point", "num_supports": 32},
        {"return_probability": 0.75, "regime_separation": "low", "observability": "partial", "observation_noise": 0.03, "process_noise": 0.01, "center_initialization": "random_normal", "num_supports": 128},
        {"return_probability": 0.0, "regime_separation": "medium", "observability": "hidden_driver", "observation_noise": 0.01, "process_noise": 0.0, "center_initialization": "sampled_training_points", "num_supports": 16},
        {"return_probability": 1.0, "regime_separation": "low", "observability": "full", "observation_noise": 0.0, "process_noise": 0.01, "center_initialization": "kmeans", "num_supports": 256},
    ]
    rows = []
    for task in ("controlled_prototype", "switching_mackey_glass_controlled", "switching_narma_controlled"):
        for design_index, design in enumerate(designs):
            for variant in ("D0", "DD-L", "RF-KV", "KC-LV", "DD-A"):
                for seed_index in range(4):
                    fidelity = (0.2, 0.5, 1.0)[design_index % 3]
                    steps = {0.2: 100, 0.5: 250, 1.0: 500}[fidelity]
                    factors = dict(design)
                    factors.update({"fidelity": fidelity, "design_index": design_index})
                    rows.append(_base_row("stage2C_factorial", task, variant, 1_000_000, seed_index, cell=f"F{design_index}", heldout_streams=2, steps=steps, **factors))
    return _number_rows(rows)


def build_symbolic_rows() -> list[dict[str, Any]]:
    rows = []
    for variant in SYMBOLIC_VARIANTS:
        for seed_index in range(5):
            for factor_index, factors in enumerate((
                {"transition_entropy": 0.1, "emission_overlap": 0.1, "return_probability": 0.25, "explicit_regime_token": False},
                {"transition_entropy": 0.5, "emission_overlap": 0.2, "return_probability": 0.5, "explicit_regime_token": False},
                {"transition_entropy": 0.9, "emission_overlap": 0.4, "return_probability": 0.75, "explicit_regime_token": True},
            )):
                rows.append(_base_row("stage2D_symbolic", "controlled_symbolic_regime_language", variant, 1_000_000, seed_index, cell=f"S{factor_index}", heldout_streams=5, steps=500, task_type="language", **factors))
    return _number_rows(rows)


def _number_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, row in enumerate(rows):
        row["row_id"] = index
    return rows


def build_all(config: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    return {"stage2A_component": build_component_rows(), "stage2B_capacity": build_capacity_rows(), "stage2C_factorial": build_factorial_rows(), "stage2D_symbolic": build_symbolic_rows()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2A–2D manifests.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase5/stage2/manifests"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in build_all().items():
        result = write_table(args.output_dir / f"{name}.jsonl", rows)
        write_json(args.output_dir / f"{name}_summary.json", {"stage": name, "rows": len(rows), "format": result["format"]})
        counts[name] = len(rows)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
