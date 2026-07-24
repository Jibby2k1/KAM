from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .table import read_table, write_json, write_table


SCALE_CONFIG: dict[str, dict[str, Any]] = {
    "XS": {"d_model": 24, "num_heads": 4, "num_layers": 1, "num_supports": 16, "seq_len": 32, "series_length": 1800, "steps": 180, "batch_size": 32, "eval_batches": 20, "target_parameters": 40_000},
    "S": {"d_model": 32, "num_heads": 4, "num_layers": 1, "num_supports": 32, "seq_len": 64, "series_length": 3000, "steps": 360, "batch_size": 32, "eval_batches": 24, "target_parameters": 250_000},
    "M": {"d_model": 48, "num_heads": 4, "num_layers": 1, "num_supports": 64, "seq_len": 128, "series_length": 5000, "steps": 700, "batch_size": 16, "eval_batches": 32, "target_parameters": 1_000_000},
    "L": {"d_model": 64, "num_heads": 4, "num_layers": 2, "num_supports": 128, "seq_len": 256, "series_length": 7000, "steps": 1000, "batch_size": 8, "eval_batches": 40, "target_parameters": 4_000_000},
}


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 2_000_000_000


def _trial_hyperparameters(trial: int, scale: str) -> dict[str, Any]:
    learning_rates = (1e-3, 3e-4, 1e-4, 6e-4)
    bandwidths = (0.5, 1.0, 2.0, 4.0)
    dropouts = (0.0, 0.05, 0.10, 0.0)
    return {
        "learning_rate": learning_rates[trial % len(learning_rates)],
        "bandwidth_init": bandwidths[(trial // len(learning_rates)) % len(bandwidths)],
        "dropout": dropouts[trial % len(dropouts)],
        "weight_decay": 1e-4 if trial % 2 == 0 else 0.0,
        "grad_clip": 1.0,
        "trial_scale_family": scale,
    }


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = list(config.get("tasks", ["switching_mackey_glass", "switching_narma", "prototype_switch"]))
    variants = list(config.get("variants", ["D0", "DD-b", "DR-b", "RF-b"]))
    scales = list(config.get("scales", ["XS", "S"]))
    trials_per_cell = int(config.get("trials_per_cell", 2))
    seeds_per_trial = int(config.get("training_seeds_per_trial", 1))
    base_seed = int(config.get("base_seed", 101))
    schedule = config.get("schedule", ["A", "B", "A"])
    if isinstance(schedule, str):
        schedule = schedule.replace("->", "-").split("-")
    precision = str(config.get("precision", "amp"))
    rows: list[dict[str, Any]] = []
    row_id = 0
    for task in tasks:
        for scale in scales:
            if scale not in SCALE_CONFIG:
                raise ValueError(f"Unknown Phase 3 scale: {scale}")
            scale_config = dict(SCALE_CONFIG[scale])
            for trial in range(trials_per_cell):
                trial_hparams = _trial_hyperparameters(trial, scale)
                for seed_slot in range(seeds_per_trial):
                    seed = base_seed + _stable_seed(task, scale, trial, seed_slot) % 1_000_000
                    for variant in variants:
                        run_id = f"p3_{task}_{variant}_{scale}_t{trial:02d}_s{seed}".replace("-", "m")
                        row = {
                            "row_id": row_id,
                            "stage": str(config.get("stage", "development_search")),
                            "task": task,
                            "variant": variant,
                            "scale": scale,
                            "trial": trial,
                            "seed_slot": seed_slot,
                            "run_id": run_id,
                            "seed": seed,
                            "schedule": list(schedule),
                            "series_length": scale_config["series_length"],
                            "seq_len": scale_config["seq_len"],
                            "batch_size": scale_config["batch_size"],
                            "steps": scale_config["steps"],
                            "eval_batches": scale_config["eval_batches"],
                            "d_model": scale_config["d_model"],
                            "num_heads": scale_config["num_heads"],
                            "num_layers": scale_config["num_layers"],
                            "num_supports": scale_config["num_supports"],
                            "max_seq_len": scale_config["seq_len"],
                            "parameter_match_target": scale_config["target_parameters"],
                            "precision": precision,
                            "memory_output": "both",
                            "route_features": "raw",
                            "expose_memory_weights": variant not in {"D0"},
                            "save_validation_predictions": trial == 0 and seed_slot == 0,
                            "causal_probe": variant in {"DD-b", "DR-b", "DD-b-staged", "DR-b-staged", "RF-b"} and trial == 0 and seed_slot == 0,
                            "online_eval": trial == 0 and seed_slot == 0,
                            "use_final_checkpoint_for_diagnostics": bool(config.get("use_final_checkpoint_for_diagnostics", False)),
                            "online_length": int(config.get("online_length", min(scale_config["series_length"], 2400))),
                            "deletion_draws": int(config.get("deletion_draws", 8)),
                            "memory_trace": bool(config.get("memory_trace", False)),
                            "memory_warmup_fraction": float(config.get("memory_warmup_fraction", 0.75)),
                            "trace_eval_every": int(config.get("trace_eval_every", 0)) or max(1, scale_config["steps"] // 10),
                            "trace_eval_batches": int(config.get("trace_eval_batches", scale_config["eval_batches"])),
                            "evaluate_train": bool(config.get("evaluate_train", False)),
                            "evaluate_test": bool(config.get("evaluate_test", False)),
                            "trace_test": bool(config.get("trace_test", True)),
                            "save_test_predictions": bool(config.get("save_test_predictions", False)),
                            **trial_hparams,
                        }
                        rows.append(row)
                        row_id += 1
    return rows


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 3 config must be a mapping")
    return payload


def build_manifest(config_path: str | Path, output_path: str | Path | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    rows = build_rows(config)
    output = Path(output_path or config.get("manifest", "results/phase3/run_manifest.jsonl"))
    if output.suffix not in {".jsonl", ".csv", ".parquet"}:
        output = output.with_suffix(".jsonl")
    result = write_table(output, rows)
    summary = {
        "config": str(config_path),
        "row_count": len(rows),
        "tasks": sorted({row["task"] for row in rows}),
        "variants": sorted({row["variant"] for row in rows}),
        "scales": sorted({row["scale"] for row in rows}),
        "format": result["format"],
        "path": result["path"],
    }
    write_json(output.with_name("manifest_summary.json"), summary)
    return rows


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = read_table(path)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    for index, row in enumerate(rows):
        row.setdefault("row_id", index)
        if isinstance(row.get("schedule"), str):
            try:
                row["schedule"] = json.loads(row["schedule"])
            except json.JSONDecodeError:
                row["schedule"] = row["schedule"].replace("->", "-").split("-")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static Phase 3 experiment manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = build_manifest(args.config, args.output)
    print(json.dumps({"rows": len(rows), "manifest": str(args.output or "results/phase3/run_manifest.jsonl")}, indent=2))


if __name__ == "__main__":
    main()
