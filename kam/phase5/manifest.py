"""Build the Phase V validity-gate manifest."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
from kam.phase4.table import write_json, write_table

VALIDITY_SCALE = {"V": {"d_model": 16, "num_heads": 4, "num_layers": 1, "num_supports": 16, "seq_len": 16, "series_length": 240, "steps": 8, "batch_size": 8, "eval_batches": 4, "target_parameters": None}}

def _seed(*parts: Any) -> int:
    return 9001 + int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % 100000

def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = list(config.get("tasks", ["controlled_prototype", "controlled_symbolic_regime_language", "switching_mackey_glass_controlled", "switching_narma_controlled"]))
    variants = list(config.get("variants", ["D0", "DD-L", "RF-FULL"]))
    protocols = list(config.get("training_protocols", ["iid_window_training", "ordered_stream_training"]))
    rows = []
    for task in tasks:
        for protocol in protocols:
            for variant in variants:
                base = VALIDITY_SCALE["V"]
                seed = _seed(task, protocol)
                run_id = f"p5_validity_{task}_{protocol}_{variant}_V_s{seed}".replace("-", "m")
                rows.append({
                    "row_id": len(rows), "stage": "stage0_validity_gate", "task": task, "variant": variant,
                    "scale": "V", "training_protocol": protocol, "run_id": run_id, "seed": seed, **base,
                    "precision": str(config.get("precision", "amp")), "route_features": "projected", "route_projection_dim": 64,
                    "memory_output": "both", "expose_memory_weights": variant != "D0", "memory_trace": False,
                    "evaluate_train": True, "evaluate_test": True, "trace_test": True,
                    "save_validation_predictions": False, "save_test_predictions": False,
                    "learning_rate": 0.0003, "weight_decay": 0.0001, "regime_count": 2,
                    "regime_separation": "medium", "return_probability": 0.5, "dwell_length": 32,
                    "transition_type": "abrupt", "observation_noise": 0.0, "process_noise": 0.0,
                    "input_noise": 0.0, "observability": "full",
                })
    return rows

def load_config(path: str | Path) -> dict[str, Any]:
    import yaml
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase V config must be a mapping")
    return payload

def build_manifest(config_path: str | Path, output_path: str | Path | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    rows = build_rows(config)
    output = Path(output_path or config.get("manifest", "results/phase5/manifests/validity.jsonl"))
    result = write_table(output, rows)
    write_json(output.with_name("manifest_summary.json"), {"config": str(config_path), "row_count": len(rows), "format": result["format"], "path": result["path"], "stage": "validity_gate", "scientific_scope": "precondition checks only; full Phase V campaign is gated"})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase V validity manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = build_manifest(args.config, args.output)
    print(json.dumps({"rows": len(rows), "manifest": str(args.output or "results/phase5/manifests/validity.jsonl")}, indent=2))

if __name__ == "__main__":
    main()
