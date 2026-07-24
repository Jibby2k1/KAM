import csv

import torch

from kam.factory import make_model
from kam.phase3.memory_trace import (
    memory_bank_parameters,
    set_memory_bank_trainable,
    snapshot_memory_bank,
    trace_row,
)
from kam.run_suite import _run_one


def _spec(variant: str) -> dict[str, object]:
    return {
        "model_name": variant,
        "task_type": "regression",
        "input_dim": 2,
        "output_dim": 1,
        "d_model": 16,
        "num_heads": 4,
        "num_layers": 1,
        "num_supports": 8,
        "max_seq_len": 12,
        "memory_output": "both",
        "expose_memory_weights": True,
    }


def test_staged_alias_preserves_architecture_and_parameter_count() -> None:
    joint = make_model(_spec("DD-b"))
    staged = make_model(_spec("DD-b-staged"))
    assert sum(parameter.numel() for parameter in joint.parameters()) == sum(parameter.numel() for parameter in staged.parameters())
    assert all(parameter.requires_grad for parameter in memory_bank_parameters(staged).values())


def test_memory_trace_reports_zero_movement_after_freeze() -> None:
    model = make_model(_spec("DD-b-staged"))
    inputs = torch.randn(2, 12, 2)
    initial = snapshot_memory_bank(model)
    first, support_rows = trace_row(model, initial, None, step=1, stage="memory_adaptation", memory_bank_trainable=True, probe_inputs=inputs)
    set_memory_bank_trainable(model, False)
    previous = snapshot_memory_bank(model)
    second, _ = trace_row(model, initial, previous, step=2, stage="backbone_finetuning", memory_bank_trainable=False, probe_inputs=inputs)
    assert first["memory_bank_trainable"] is True
    assert second["memory_bank_trainable"] is False
    assert second["memory_key_step_delta"] == 0.0
    assert second["memory_value_step_delta"] == 0.0
    assert len(support_rows) == 8


def test_run_one_writes_staged_trace_and_final_test(tmp_path) -> None:
    run = {
        "run_id": "phase3_staged_unit",
        "task": "prototype_switch",
        "variant": "DD-b-staged",
        "seed": 17,
        "series_length": 240,
        "seq_len": 12,
        "batch_size": 8,
        "steps": 6,
        "eval_batches": 2,
        "trace_eval_every": 2,
        "trace_eval_batches": 2,
        "d_model": 16,
        "num_heads": 4,
        "num_layers": 1,
        "num_supports": 8,
        "max_seq_len": 12,
        "memory_output": "both",
        "expose_memory_weights": True,
        "memory_trace": True,
        "evaluate_train": True,
        "evaluate_test": True,
        "trace_test": True,
        "save_validation_predictions": False,
        "save_test_predictions": False,
    }
    metrics = _run_one(run, tmp_path, torch.device("cpu"), "fp32")
    run_dir = tmp_path / run["run_id"]
    assert metrics["memory_protocol"] == "warmup_then_freeze"
    assert metrics["final_test"]["mse"] >= 0.0
    assert (run_dir / "final_model.pt").exists()
    with (run_dir / "memory_training_trace.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["stage"] == "memory_adaptation" for row in rows)
    assert any(row["stage"] == "backbone_finetuning" for row in rows)
    post_freeze = [row for row in rows if row["stage"] == "backbone_finetuning"]
    assert all(float(row["memory_key_step_delta"]) == 0.0 for row in post_freeze)
    assert all(float(row["memory_value_step_delta"]) == 0.0 for row in post_freeze)
