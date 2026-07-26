from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class AlternatingSchedule:
    """Explicit algebra:geometry schedule used by every optimizer ablation."""

    mode: str = "joint"
    algebra_steps: int = 1
    geometry_steps: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"joint", "alternating", "solve_algebra", "variable_projection"}:
            raise ValueError("unknown optimization mode")
        if self.algebra_steps <= 0 or self.geometry_steps <= 0:
            raise ValueError("schedule step counts must be positive")

    @classmethod
    def from_label(cls, label: str) -> "AlternatingSchedule":
        if label == "joint":
            return cls("joint")
        if label == "full_solve":
            return cls("solve_algebra", algebra_steps=1, geometry_steps=1)
        if label.startswith("alternating_"):
            ratio = int(label.split("_", 1)[1].split(":")[0])
            return cls("alternating", algebra_steps=ratio, geometry_steps=1)
        if label == "variable_projection":
            return cls("variable_projection")
        raise ValueError(f"unknown schedule label: {label}")

    def phase(self, step: int) -> str:
        if self.mode in {"joint", "solve_algebra", "variable_projection"}:
            return "joint"
        cycle = self.algebra_steps + self.geometry_steps
        return "algebra" if step % cycle < self.algebra_steps else "geometry"


@dataclass
class TrustRegionState:
    radius: float = 1.0
    accepted: int = 0
    rejected: int = 0

    def propose(self, current: Tensor, candidate: Tensor) -> tuple[Tensor, bool, float]:
        delta = candidate - current
        norm = float(delta.norm().detach())
        if not torch.isfinite(candidate).all() or norm > self.radius:
            self.rejected += 1
            return current.clone(), False, norm
        self.accepted += 1
        return candidate.clone(), True, norm

    def update(self, accepted: bool) -> None:
        if accepted:
            self.radius *= 1.05
        else:
            self.radius *= 0.5


class AlternatingOptimizer:
    """Callback-based alternating loop with explicit algebra/geometry phases."""

    def __init__(self, schedule: AlternatingSchedule, trust_region: TrustRegionState | None = None) -> None:
        self.schedule = schedule
        self.trust_region = trust_region or TrustRegionState()

    def run(
        self,
        outer_iterations: int,
        *,
        optimize_algebra: Callable[[int], dict[str, Any] | None],
        update_geometry: Callable[[int, float], tuple[bool, dict[str, Any] | None]],
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for outer in range(outer_iterations):
            algebra_result = optimize_algebra(outer)
            accepted, geometry_result = update_geometry(outer, self.trust_region.radius)
            self.trust_region.update(accepted)
            history.append({
                "outer_iteration": outer,
                "phase": self.schedule.phase(outer),
                "algebra": algebra_result or {},
                "geometry": geometry_result or {},
                "geometry_accepted": accepted,
                "trust_radius": self.trust_region.radius,
            })
        return history
