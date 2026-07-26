from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kam.memory.drift import feature_drift, function_drift


@dataclass
class TrustRegionDecision:
    accepted: bool
    reason: str
    feature_drift: float
    function_drift: float
    support_utilization: float
    condition_number: float


class GeometryTrustRegion:
    """Anchor-set acceptance/rollback policy for slow geometry updates."""

    def __init__(self, max_feature_drift: float = 0.25, max_function_drift: float = 0.25, max_condition_number: float = 1e8, min_support_utilization: float = 0.05) -> None:
        self.max_feature_drift = max_feature_drift
        self.max_function_drift = max_function_drift
        self.max_condition_number = max_condition_number
        self.min_support_utilization = min_support_utilization

    def evaluate(
        self,
        old_features: Tensor,
        new_features: Tensor,
        old_output: Tensor,
        new_output: Tensor,
        *,
        objective_old: float,
        objective_new: float,
        support_utilization: float,
        condition_number: float,
    ) -> TrustRegionDecision:
        feature_change = feature_drift(new_features, old_features)
        function_change = function_drift(new_output, old_output)
        checks = [
            (objective_new <= objective_old, "objective_increase"),
            (feature_change <= self.max_feature_drift, "feature_drift"),
            (function_change <= self.max_function_drift, "function_drift"),
            (support_utilization >= self.min_support_utilization, "support_collapse"),
            (condition_number <= self.max_condition_number, "conditioning"),
        ]
        for passed, reason in checks:
            if not passed:
                return TrustRegionDecision(False, reason, feature_change, function_change, support_utilization, condition_number)
        return TrustRegionDecision(True, "accepted", feature_change, function_change, support_utilization, condition_number)


__all__ = ["GeometryTrustRegion", "TrustRegionDecision"]
