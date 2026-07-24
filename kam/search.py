from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import optuna

from .run_suite import load_config, run_suite


def _sample(trial: optuna.Trial, name: str, value: Any) -> Any:
    if isinstance(value, list):
        return trial.suggest_categorical(name, value)
    if isinstance(value, dict) and value.get("type") == "log_float":
        return trial.suggest_float(name, float(value["low"]), float(value["high"]), log=True)
    if isinstance(value, dict) and value.get("type") == "float":
        return trial.suggest_float(name, float(value["low"]), float(value["high"]))
    return value


def _objective(
    trial: optuna.Trial,
    *,
    base_config: dict[str, Any],
    family: str,
    task: str,
    seed: int,
    output_root: Path,
    max_vram_gb: float | None,
) -> float:
    search = base_config.get("search", {})
    space = search.get("space", {})
    sampled = {name: _sample(trial, name, value) for name, value in space.items()}
    if int(sampled.get("d_model", 48)) % int(sampled.get("num_heads", 4)):
        raise optuna.TrialPruned("num_heads must divide d_model")
    context_window = sampled.get("context_window")
    if context_window == "full":
        context_window = None
    run = {
        "run_id": f"trial_{family}_{task}_{trial.number:04d}",
        "variant": family,
        "task": task,
        "seed": seed + trial.number,
        "d_model": int(sampled.get("d_model", 48)),
        "num_heads": int(sampled.get("num_heads", 4)),
        "num_layers": int(sampled.get("num_layers", 1)),
        "num_supports": int(sampled.get("num_supports", 32)),
        "context_window": context_window,
        "ffn_expansion": int(sampled.get("ffn_expansion", 4)),
        "learning_rate": float(sampled.get("learning_rate", 3e-4)),
        "weight_decay": float(sampled.get("weight_decay", 1e-4)),
        "dropout": float(sampled.get("dropout", 0.0)),
        "bandwidth_init": float(sampled.get("bandwidth_init", 1.0)),
        "bandwidth": "learned" if bool(sampled.get("bandwidth_learned", True)) else "fixed",
        "route_projection_dim": sampled.get("route_projection_dim"),
        "series_length": int(base_config.get("search_series_length", 4000)),
        "seq_len": int(base_config.get("search_seq_len", 32)),
        "batch_size": int(base_config.get("search_batch_size", 32)),
        "steps": int(base_config.get("search_steps", 50)),
    }
    if task.startswith("switching_"):
        run["schedule"] = base_config.get("search_schedule", ["A", "B", "A"])
    trial_root = output_root / family / task
    trial_config = {
        "suite_name": f"{base_config.get('suite_name', 'phase2')}_optuna_{family}_{task}",
        "device": base_config.get("device", "auto"),
        "precision": base_config.get("precision", "amp"),
        "output_root": str(trial_root),
        "study_database": str(trial_root / "run_registry.sqlite"),
        "metrics_table": str(trial_root / "all_metrics.csv"),
        "fail_fast": True,
        "runs": [run],
    }
    started = time.perf_counter()
    result = run_suite(trial_config)
    if not result:
        raise RuntimeError("Optuna trial produced no completed run")
    metrics = result[-1]
    validation = metrics.get("final_validation", {})
    objective = float(validation.get("mse", validation.get("cross_entropy", float("inf"))))
    trial.set_user_attr("run_id", run["run_id"])
    trial.set_user_attr("parameters", metrics.get("parameter_count"))
    trial.set_user_attr("seconds", time.perf_counter() - started)
    trial.report(objective, step=int(run["steps"]))
    if trial.should_prune():
        raise optuna.TrialPruned()
    if max_vram_gb is not None and float(metrics.get("peak_memory_megabytes", 0.0)) > max_vram_gb * 1024.0:
        raise optuna.TrialPruned("peak VRAM limit exceeded")
    return objective


def run_search(
    config: dict[str, Any],
    *,
    trials_per_family_task: int,
    seed: int,
    families: list[str] | None = None,
    tasks: list[str] | None = None,
    output_root: Path | None = None,
    max_vram_gb: float | None = None,
) -> list[dict[str, Any]]:
    search = config.get("search", {})
    families = families or list(search.get("families", ["D0", "DD", "DR", "RR"]))
    tasks = tasks or list(config.get("search_tasks", ["mackey_glass", "narma"]))
    output_root = output_root or Path(config.get("output_root", "results/phase2")) / "optuna_search"
    output_root.mkdir(parents=True, exist_ok=True)
    storage_path = Path(config.get("search_database", output_root / "optuna.sqlite"))
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for family in families:
        for task in tasks:
            study_name = f"{config.get('suite_name', 'phase2')}_{family}_{task}"
            sampler = optuna.samplers.TPESampler(seed=seed)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1)
            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{storage_path}",
                load_if_exists=True,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
            )
            study.optimize(
                lambda trial: _objective(
                    trial,
                    base_config=config,
                    family=family,
                    task=task,
                    seed=seed,
                    output_root=output_root,
                    max_vram_gb=max_vram_gb,
                ),
                n_trials=trials_per_family_task,
            )
            best = study.best_trial if study.best_trial is not None else None
            summaries.append({
                "study": study_name,
                "family": family,
                "task": task,
                "trials": len(study.trials),
                "complete_trials": sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials),
                "pruned_trials": sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials),
                "best_value": best.value if best else None,
                "best_trial": best.number if best else None,
            })
    with (output_root / "search_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (output_root / "search_config.json").write_text(json.dumps({"config": config, "families": families, "tasks": tasks, "trials_per_family_task": trials_per_family_task}, default=str, indent=2), encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Optuna SQLite-backed Phase II search with pruning.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-vram-gb", type=float, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    trials = int(args.trials or config.get("search", {}).get("trials_per_family_task", 32))
    summaries = run_search(config, trials_per_family_task=trials, seed=args.seed, families=args.families, tasks=args.tasks, output_root=args.output_root, max_vram_gb=args.max_vram_gb)
    print(f"Completed {len(summaries)} Optuna studies.")


if __name__ == "__main__":
    main()
