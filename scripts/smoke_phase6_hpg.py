#!/usr/bin/env python3
"""Run representative Phase 6 rows inside a real HPG GPU allocation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kam.phase6.run_array import run_row


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Phase 6 GPU smoke must run inside a CUDA allocation")
    checks: list[tuple[str, dict]] = []
    for architecture in ("T0", "T-WIDE", "T-MEMTOK", "T-MOE", "T-PKM", "T-KAM-F", "T-KAM-L", "T-KAM-ALT", "T-KAM-VP"):
        checks.append((f"transformer_{architecture}", {
            "stage": "stage2_transformer_comparison",
            "stage_mode": "profile",
            "row_id": f"smoke_{architecture}",
            "seed": 700001 + len(checks),
            "task": "mqar",
            "architecture": architecture,
            "scale": "2M",
            "target_parameter_budget": 2_000_000,
            "num_supports": 32,
            "top_k": 4,
        }))
    for router in ("exact", "chunked", "approximate", "product_key"):
        checks.append((f"router_{router}", {
            "stage": "stage3_router_scaling",
            "stage_mode": "profile",
            "row_id": f"smoke_router_{router}",
            "seed": 710001 + len(checks),
            "router": router,
            "slot": 1000,
            "top_k": 4,
            "precision": "fp32",
        }))
    for architecture, adapter in (("T0", "none"), ("T-WIDE", "nlms"), ("T-KAM-L", "slow_geometry"), ("T-KAM-ONLINE", "episodic_insertion"), ("T-KAM-DUAL", "episodic_insertion")):
        checks.append((f"online_{architecture}_{adapter}", {
            "stage": "stage4_online_adaptation",
            "stage_mode": "profile",
            "row_id": f"smoke_online_{architecture}_{adapter}",
            "seed": 720001 + len(checks),
            "task": "symbolic_schedule",
            "architecture": architecture,
            "adapter": adapter,
            "num_supports": 32,
            "top_k": 4,
        }))
    checks.append(("long_T-KAM-L", {
        "stage": "stage5_long_training",
        "stage_mode": "profile",
        "row_id": "smoke_long_T-KAM-L",
        "seed": 730001,
        "task": "small_language",
        "architecture": "T-KAM-L",
        "scale": "30M",
        "target_parameter_budget": 30_000_000,
        "token_budget": 200_000_000,
        "training_token_cap": 4096,
        "num_supports": 32,
        "top_k": 4,
    }))
    checks.append(("confirmation_ALT", {
        "stage": "stage6_confirmation",
        "stage_mode": "profile",
        "row_id": "smoke_confirmation_ALT",
        "seed": 740001,
        "task": "prototype",
        "claim": "alternating_vs_joint",
        "architecture": "T-KAM-ALT",
        "scale": "10M",
        "target_parameter_budget": 10_000_000,
        "num_supports": 32,
        "top_k": 4,
    }))
    for label, row in checks:
        result = run_row(row, device="cuda")
        metrics = result.get("metrics", {})
        print(json.dumps({
            "label": label,
            "status": result.get("status"),
            "error": result.get("error"),
            "total_parameters": metrics.get("total_parameters"),
            "parameter_match_error_fraction": metrics.get("parameter_match_error_fraction"),
            "recall_at_k_against_exact": metrics.get("recall_at_k_against_exact"),
            "training_optimizer_mode": metrics.get("training_optimizer_mode"),
            "geometry_update_steps": metrics.get("geometry_update_steps"),
            "episodic_active": metrics.get("episodic_active"),
            "training_tokens": metrics.get("training_tokens"),
        }, sort_keys=True))
        if result.get("status") != "pass":
            raise SystemExit(f"GPU smoke failed: {label}: {result.get('error')}")


if __name__ == "__main__":
    main()
