"""Build the bounded Stage 1 Phase V mechanism-pilot manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from kam.phase4.table import write_json, write_table

PILOT_SCALES: dict[str, dict[str, Any]] = {
    "P250": {
        "target_active_parameters": 250_000,
        "d0": {"d_model": 80, "num_heads": 4, "num_layers": 3, "num_supports": 64},
        "bank": {"d_model": 72, "num_heads": 4, "num_layers": 3, "num_supports": 16},
        "fourier_features": 1923,
    },
    "P1M": {
        "target_active_parameters": 1_000_000,
        "d0": {"d_model": 164, "num_heads": 4, "num_layers": 3, "num_supports": 64},
        "bank": {"d_model": 136, "num_heads": 4, "num_layers": 3, "num_supports": 128},
        "fourier_features": 7692,
    },
}

TASKS = (
    "controlled_prototype",
    "controlled_symbolic_regime_language",
    "switching_mackey_glass_controlled",
    "switching_narma_controlled",
)
VARIANTS = ("D0", "DD-L", "RF-KV", "RF-FULL", "KC-LV", "RFF")


def _seed(task: str, scale: str, seed_index: int) -> int:
    payload = f"phase5-pilot|{task}|{scale}|{seed_index}".encode()
    return 12001 + int(hashlib.sha256(payload).hexdigest()[:8], 16) % 100000


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = tuple(config.get("tasks", TASKS))
    variants = tuple(config.get("variants", VARIANTS))
    scales = tuple(config.get("scales", PILOT_SCALES))
    seeds = int(config.get("seeds", 3))
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for scale in scales:
            if scale not in PILOT_SCALES:
                raise ValueError(f"unsupported pilot scale: {scale}")
            preset = PILOT_SCALES[scale]
            for variant in variants:
                for seed_index in range(seeds):
                    architecture = preset["d0"] if variant == "D0" else preset["bank"]
                    run_id = f"p5_pilot_{task}_{scale}_{variant}_s{seed_index}".replace("-", "m")
                    row = {
                        "row_id": len(rows),
                        "stage": "stage1_mechanism_pilot",
                        "task": task,
                        "variant": variant,
                        "scale": scale,
                        "seed_index": seed_index,
                        "seed": _seed(task, scale, seed_index),
                        "run_id": run_id,
                        "target_active_parameters": preset["target_active_parameters"],
                        "active_match_tolerance": 0.01,
                        "training_protocol": str(config.get("training_protocol", "iid_window_training")),
                        "d_model": architecture["d_model"],
                        "num_heads": architecture["num_heads"],
                        "num_layers": architecture["num_layers"],
                        "num_supports": architecture["num_supports"],
                        "fourier_features": preset["fourier_features"] if variant == "RFF" else None,
                        "seq_len": 64,
                        "series_length": 640,
                        "train_length": 640,
                        "validation_length": 256,
                        "test_length": 256,
                        "prequential_length": 256,
                        "steps": int(config.get("steps", 256)),
                        "eval_every": 32,
                        "batch_size": int(config.get("batch_size", 32)),
                        "eval_batches": int(config.get("eval_batches", 16)),
                        "trace_eval_every": 64,
                        "trace_eval_batches": 4,
                        "learning_rate": 3e-4,
                        "weight_decay": 1e-4,
                        "precision": str(config.get("precision", "amp")),
                        "route_features": "projected",
                        "route_projection_dim": 64,
                        "memory_output": "both",
                        "expose_memory_weights": variant not in {"D0", "RFF"},
                        "memory_trace": variant not in {"D0", "RFF"},
                        "trace_test": True,
                        "evaluate_train": True,
                        "evaluate_test": True,
                        "save_validation_predictions": False,
                        "save_test_predictions": True,
                        "regime_count": 2,
                        "regime_separation": "medium",
                        "return_probability": 0.5,
                        "dwell_length": 64,
                        "transition_type": "abrupt",
                        "observation_noise": 0.0,
                        "process_noise": 0.0,
                        "input_noise": 0.0,
                        "observability": "full",
                    }
                    rows.append(row)
    return rows


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pilot config must be a mapping")
    return payload


def build_manifest(config_path: str | Path, output_path: str | Path | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    rows = build_rows(config)
    output = Path(output_path or config.get("manifest", "results/phase5/manifests/pilot.jsonl"))
    result = write_table(output, rows)
    write_json(output.with_name("pilot_manifest_summary.json"), {
        "config": str(config_path),
        "row_count": len(rows),
        "format": result["format"],
        "path": result["path"],
        "stage": "stage1_mechanism_pilot",
        "scientific_scope": "signal and variance profiling; not confirmatory evidence",
        "scales": PILOT_SCALES,
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase V Stage 1 pilot manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = build_manifest(args.config, args.output)
    print(json.dumps({"rows": len(rows), "manifest": str(args.output or "results/phase5/manifests/pilot.jsonl")}, indent=2))


if __name__ == "__main__":
    main()
