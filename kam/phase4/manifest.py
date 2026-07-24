"""Build the bounded Phase IV data-regime factorial screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .table import write_json, write_table


SCALE_CONFIG: dict[str, dict[str, Any]] = {
    "S": {"d_model": 32, "num_heads": 4, "num_layers": 1, "num_supports": 32, "seq_len": 64, "series_length": 3000, "steps": 240, "batch_size": 32, "eval_batches": 24, "target_parameters": 250_000},
    "M": {"d_model": 48, "num_heads": 4, "num_layers": 1, "num_supports": 64, "seq_len": 128, "series_length": 5000, "steps": 480, "batch_size": 16, "eval_batches": 32, "target_parameters": 1_000_000},
}


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 2_000_000_000


def _condition(task: str, condition: str) -> dict[str, Any]:
    """Return a controlled, currently-supported data setting."""
    if task == "prototype_switch":
        return {"schedule": ["A", "B", "A"] if condition == "recurring" else ["A", "B", "A", "B"]}
    if task == "switching_narma":
        scale = 1.05 if condition == "recurring" else 1.15
        return {"schedule": ["A", "B", "A"], "regimes": {
            "A": {"order": 10, "coefficient_scale": 1.0, "noise_std": 0.0},
            "B": {"order": 10, "coefficient_scale": scale, "noise_std": 0.0},
            "C": {"order": 20, "coefficient_scale": 1.0, "noise_std": 0.0},
        }}
    if task == "switching_mackey_glass":
        tau_b = 18.0 if condition == "recurring" else 20.0
        return {"schedule": ["A", "B", "A"], "regimes": {
            "A": {"tau": 17.0, "beta": 0.20},
            "B": {"tau": tau_b, "beta": 0.20},
            "C": {"tau": 17.0, "beta": 0.22},
        }}
    raise ValueError(f"Unsupported Phase IV task: {task}")


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = list(config.get("tasks", ["prototype_switch", "switching_narma", "switching_mackey_glass"]))
    variants = list(config.get("variants", ["D0", "DD-b", "DD-b-staged", "RF-b"]))
    scales = list(config.get("scales", ["S", "M"]))
    conditions = list(config.get("conditions", ["recurring", "separated"]))
    seeds_per_cell = int(config.get("seeds_per_cell", 2))
    base_seed = int(config.get("base_seed", 7401))
    rows: list[dict[str, Any]] = []
    row_id = 0
    for task in tasks:
        for condition in conditions:
            factors = _condition(task, condition)
            for scale in scales:
                if scale not in SCALE_CONFIG:
                    raise ValueError(f"Unknown Phase IV scale: {scale}")
                scale_config = SCALE_CONFIG[scale]
                for seed_slot in range(seeds_per_cell):
                    seed = base_seed + _stable_seed(task, condition, scale, seed_slot) % 1_000_000
                    for variant in variants:
                        run_id = f"p4_{task}_{condition}_{variant}_{scale}_s{seed}".replace("-", "m")
                        rows.append({
                            "row_id": row_id, "stage": "B_factorial_mechanism_screen", "task": task,
                            "condition": condition, "factor_definition": "recurrence/dwell, coefficient separation, or delay separation",
                            "variant": variant, "scale": scale, "seed_slot": seed_slot, "run_id": run_id, "seed": seed,
                            **factors, **scale_config, "max_seq_len": scale_config["seq_len"],
                            "parameter_match_target": scale_config["target_parameters"], "precision": str(config.get("precision", "amp")),
                            "memory_output": "both", "route_features": "raw", "expose_memory_weights": variant != "D0",
                            "memory_trace": variant in {"DD-b", "DD-b-staged", "RF-b"},
                            "memory_warmup_fraction": float(config.get("memory_warmup_fraction", 0.75)),
                            "trace_eval_every": int(config.get("trace_eval_every", 0)) or max(1, scale_config["steps"] // 12),
                            "trace_eval_batches": scale_config["eval_batches"], "evaluate_train": True, "evaluate_test": True,
                            "trace_test": True, "save_validation_predictions": True, "save_test_predictions": True,
                            "learning_rate": float(config.get("learning_rate", 3e-4)), "weight_decay": float(config.get("weight_decay", 1e-4)), "grad_clip": 1.0,
                        })
                        row_id += 1
    return rows


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase IV config must be a mapping")
    return payload


def build_manifest(config_path: str | Path, output_path: str | Path | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    rows = build_rows(config)
    output = Path(output_path or config.get("manifest", "results/phase4/factorial_screen.jsonl"))
    result = write_table(output, rows)
    write_json(output.with_name("manifest_summary.json"), {
        "config": str(config_path), "row_count": len(rows), "tasks": sorted({row["task"] for row in rows}),
        "conditions": sorted({row["condition"] for row in rows}), "variants": sorted({row["variant"] for row in rows}),
        "scales": sorted({row["scale"] for row in rows}), "format": result["format"], "path": result["path"],
        "scientific_scope": "bounded Stage B screen; not the locked confirmatory campaign",
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase IV factorial screen manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = build_manifest(args.config, args.output)
    print(json.dumps({"rows": len(rows), "manifest": str(args.output or "results/phase4/factorial_screen.jsonl")}, indent=2))


if __name__ == "__main__":
    main()
