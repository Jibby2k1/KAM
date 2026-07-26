from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TASKS = (
    "route_exact_reference",
    "route_chunked_reference",
    "gradient_finite",
    "zero_gate_equivalence",
    "resource_accounting",
    "ridge_solver",
    "geometry_rollback",
    "causal_mask",
)

TRANSFORMER_PARAMETER_TARGETS = {
    "2M": 2_000_000,
    "10M": 10_000_000,
    "30M": 30_000_000,
    "100M": 100_000_000,
}

STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "stage1_mechanism": {
        "target_jobs": 3000,
        "profile_jobs": 64,
        "tasks": ["prototype", "switching_mackey_glass", "switching_narma", "mqar"],
        "architectures": ["T0", "T-WIDE", "T-MEMTOK", "T-KAM-F", "T-KAM-L"],
        "optimizers": ["joint_sgd", "alternating_8_1", "alternating_32_1", "alternating_128_1", "ridge_resolve", "variable_projection_stopgrad", "variable_projection_implicit", "dictionary_update"],
        "geometries": ["fixed_random", "fixed_data_sample", "fixed_kmeans", "fixed_farthest_point", "learned_full", "learned_low_rank_delta"],
        "experts": ["vector", "low_rank_affine_expert", "routes_only"],
        "supports": [16, 32, 64, 128, 256, 512],
        "top_k": [1, 2, 4, 8, 16],
        "fidelities": [0.05, 0.2, 0.5, 1.0],
    },
    "stage2_transformer_comparison": {
        "target_jobs": 1500,
        "profile_jobs": 48,
        "tasks": ["mqar", "controlled_symbolic_regimes", "small_language", "prototype"],
        "architectures": ["T0", "T-WIDE", "T-MEMTOK", "T-MOE", "T-PKM", "T-KAM-F", "T-KAM-L", "T-KAM-ALT", "T-KAM-VP"],
        "scales": ["2M", "10M", "30M", "100M"],
        "profile_scales": ["2M", "10M", "30M"],
        "seeds": [1, 2, 3],
    },
    "stage3_router_scaling": {
        "target_jobs": 500,
        "profile_jobs": 32,
        "tasks": ["router_scaling"],
        "routers": ["exact", "chunked", "product_key", "approximate"],
        "slots": [1000, 4000, 16000, 64000, 262000, 1000000],
        "top_k": [1, 2, 4, 8, 16, 32],
        "precisions": ["fp32", "bf16", "fp16"],
    },
    "stage4_online_adaptation": {
        "target_jobs": 1000,
        "profile_jobs": 48,
        "tasks": ["mackey_glass_schedule", "narma_schedule", "prototype_schedule", "symbolic_schedule"],
        "architectures": ["T0", "T-WIDE", "T-KAM-F", "T-KAM-L", "T-KAM-ONLINE", "T-KAM-DUAL"],
        "adapters": ["none", "sgd", "nlms", "rls", "value_only", "expert_only", "episodic_insertion", "slow_geometry"],
        "seeds": [1, 2, 3, 4, 5],
    },
    "stage5_long_training": {
        "target_jobs": 100,
        "profile_jobs": 12,
        "tasks": ["small_language", "prototype", "switching_mackey_glass"],
        "architectures": ["T0", "T-MOE", "T-KAM-F", "T-KAM-L"],
        "scales": ["10M", "30M", "100M"],
        "profile_scales": ["10M", "30M"],
        "token_budgets": [200_000_000, 600_000_000, 2_000_000_000],
        "profile_token_cap": 4096,
    },
    "stage6_confirmation": {
        "target_jobs": 39,
        "profile_jobs": 12,
        "tasks": ["prototype", "switching_mackey_glass", "mqar", "small_language"],
        "claims": ["kam_vs_widened_ffn", "kam_vs_moe_pkm", "alternating_vs_joint"],
        "architectures": ["T0", "T-WIDE", "T-MOE", "T-PKM", "T-KAM-F", "T-KAM-L", "T-KAM-ALT", "T-KAM-VP"],
        "scales": ["10M"],
        "seeds": list(range(10, 17)),
    },
}


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return 700001 + int(digest[:8], 16) % 900000


def _row_id(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
    stage = str(row.get("stage", "stage0")).replace("_", "")
    return "p6" + stage[:5] + "_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def latin_hypercube(count: int, dimensions: int, seed: int = 0) -> list[list[float]]:
    """Deterministic Latin-hypercube coordinates for bounded campaign designs."""
    if count <= 0 or dimensions <= 0:
        return []
    import random

    generator = random.Random(seed)
    columns: list[list[float]] = []
    for dimension in range(dimensions):
        order = list(range(count))
        generator.shuffle(order)
        columns.append([(index + generator.random()) / count for index in order])
    return [[columns[dimension][row] for dimension in range(dimensions)] for row in range(count)]


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 6 config must be a mapping")
    return payload


def build_stage0_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    stage = config.get("stage0", {})
    tasks = list(stage.get("tasks", DEFAULT_TASKS))
    routers = list(stage.get("routers", ["exact", "chunked"]))
    experts = list(stage.get("experts", ["vector", "affine"]))
    geometries = list(stage.get("geometries", ["fixed_random", "learned_full"]))
    seeds = list(stage.get("seeds", [11, 23]))
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for router in routers:
            for expert in experts:
                for geometry in geometries:
                    for seed_tag in seeds:
                        row = {
                            "stage": "stage0_validity",
                            "task": task,
                            "router": router,
                            "expert": expert,
                            "geometry": geometry,
                            "seed_tag": seed_tag,
                            "seed": _seed("phase6", task, router, expert, geometry, seed_tag),
                            "d_model": int(stage.get("d_model", 16)),
                            "num_supports": int(stage.get("num_supports", 32)),
                            "top_k": int(stage.get("top_k", 4)),
                            "tokens": int(stage.get("tokens", 12)),
                            "router_chunk_size": int(stage.get("router_chunk_size", 7)),
                            "reference_commit": str(config.get("reference_commit", "unknown")),
                            "immutable_config": str(config.get("immutable_config", "configs/phase6/stage0_validity.yaml")),
                        }
                        row["row_id"] = _row_id(row)
                        rows.append(row)
    return rows


def _pick(values: list[Any], coordinate: float, index: int) -> Any:
    if not values:
        return None
    return values[min(len(values) - 1, int(coordinate * len(values)))]


def build_stage_rows(config: dict[str, Any], *, mode: str = "profile") -> list[dict[str, Any]]:
    """Build a static stage design without a shared database or runtime state."""
    stage_name = str(config.get("stage", ""))
    defaults = dict(STAGE_DEFAULTS.get(stage_name, {}))
    stage_config = dict(defaults)
    stage_config.update(config.get("design", {}))
    if "profile_token_cap" in config:
        stage_config["profile_token_cap"] = config["profile_token_cap"]
    if "profile_scales" in config:
        stage_config["profile_scales"] = list(config["profile_scales"])
    if mode == "profile" and stage_config.get("profile_scales"):
        stage_config["scales"] = list(stage_config["profile_scales"])
    count = int(stage_config.get("target_jobs" if mode == "full" else "profile_jobs", 0))
    if count <= 0:
        raise ValueError(f"no {mode} row count declared for {stage_name}")
    dimensions = latin_hypercube(count, 4, int(config.get("seed", 0)))
    categorical_fields = [field for field in ("tasks", "architectures", "optimizers", "geometries", "experts", "routers", "adapters", "scales", "slots", "top_k", "precisions", "fidelities", "claims", "token_budgets", "seeds") if stage_config.get(field)]
    rows: list[dict[str, Any]] = []
    for index in range(count):
        coordinate = dimensions[index]
        row: dict[str, Any] = {
            "stage": stage_name,
            "stage_mode": mode,
            "design_method": "latin_hypercube",
            "design_index": index,
            "seed": _seed("phase6", stage_name, index, config.get("seed", 0)),
            "reference_commit": str(config.get("reference_commit", "unknown")),
            "immutable_config": str(config.get("immutable_config", "unknown")),
            "d_model": int(config.get("d_model", 32)),
            "num_supports": int(config.get("num_supports", 32)),
            "top_k": 4,
            "tokens": int(config.get("tokens", 32)),
            "fidelity": 1.0,
        }
        field_names = {
            "seeds": "seed_tag",
            "slots": "slot",
            "fidelities": "fidelity",
            "token_budgets": "token_budget",
            "geometries": "geometry",
        }
        for field_index, field in enumerate(categorical_fields):
            row[field_names.get(field, field.rstrip("s"))] = _pick(list(stage_config[field]), coordinate[field_index % len(coordinate)], index)
        if "top_k" in row and isinstance(row["top_k"], str):
            row["top_k"] = int(row["top_k"])
        if "slot" in row:
            row["num_supports"] = int(row["slot"])
        if "supports" in stage_config:
            row["num_supports"] = int(_pick(list(stage_config["supports"]), coordinate[0], index))
        if "fidelities" in stage_config:
            row["fidelity"] = float(_pick(list(stage_config["fidelities"]), coordinate[1], index))
        if stage_name == "stage5_long_training" and mode == "profile":
            row["training_token_cap"] = int(stage_config.get("profile_token_cap", 4096))
        if stage_name in {"stage2_transformer_comparison", "stage5_long_training", "stage6_confirmation"} and row.get("scale") in TRANSFORMER_PARAMETER_TARGETS:
            row["target_parameter_budget"] = TRANSFORMER_PARAMETER_TARGETS[str(row["scale"])]
        row["row_id"] = _row_id(row)
        rows.append(row)
    return rows


def build_stage_manifest(config_path: str | Path, output_path: str | Path | None = None, *, mode: str = "profile") -> list[dict[str, Any]]:
    config = load_config(config_path)
    stage_name = str(config.get("stage", "stage1_mechanism"))
    if stage_name == "stage0_validity":
        return build_stage0_manifest(config_path, output_path)
    rows = build_stage_rows(config, mode=mode)
    default_output = f"results/phase6/{stage_name}/manifests/{mode}.jsonl"
    if output_path is not None:
        output = Path(output_path)
    elif mode == "profile" and config.get("manifest"):
        output = Path(config["manifest"])
    else:
        output = Path(config.get(f"{mode}_manifest", default_output))
    write_jsonl(output, rows)
    summary = {
        "stage": stage_name,
        "mode": mode,
        "row_count": len(rows),
        "target_jobs": STAGE_DEFAULTS.get(stage_name, {}).get("target_jobs"),
        "manifest": str(output),
        "config": str(config_path),
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "upstream_gate_required": "stage0_validity",
    }
    summary_text = json.dumps(summary, indent=2) + "\n"
    output.with_name("manifest_summary.json").write_text(summary_text, encoding="utf-8")
    output.with_name(f"{output.stem}_summary.json").write_text(summary_text, encoding="utf-8")
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_stage0_manifest(config_path: str | Path, output_path: str | Path | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    rows = build_stage0_rows(config)
    output = Path(output_path or config.get("manifest", "results/phase6/stage0/manifests/validity.jsonl"))
    write_jsonl(output, rows)
    summary = {
        "stage": "stage0_validity",
        "row_count": len(rows),
        "manifest": str(output),
        "config": str(config_path),
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "large_stage_submission_allowed": False,
    }
    output.with_name("manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the immutable Phase 6 Stage 0 manifest")
    parser.add_argument("--config", type=Path, default=Path("configs/phase6/stage0_validity.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("profile", "full"), default="profile")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = build_stage_manifest(args.config, args.output, mode=args.mode)
    default_output = config.get("manifest", f"results/phase6/{config.get('stage', 'stage0_validity')}/manifests/{args.mode}.jsonl")
    print(json.dumps({"rows": len(rows), "output": str(args.output or default_output), "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
