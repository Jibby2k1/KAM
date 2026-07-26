"""Algebra/geometry optimization fixtures for Phase 6."""

from .alternating import AlternatingOptimizer, AlternatingSchedule, TrustRegionState
from .algebra import ParameterPartition, algebra_transport, partition_parameters, solve_algebra
from .dictionary_update import dictionary_update, nearest_assignments
from .nlms import NLMSReadout
from .ridge import RidgeResult, ridge_objective, ridge_solve, streaming_rls
from .rls import RLSReadout, recursive_least_squares
from .trust_region import GeometryTrustRegion, TrustRegionDecision
from .variable_projection import variable_projection_objective

__all__ = [
    "AlternatingSchedule",
    "AlternatingOptimizer",
    "GeometryTrustRegion",
    "NLMSReadout",
    "ParameterPartition",
    "RidgeResult",
    "TrustRegionState",
    "TrustRegionDecision",
    "RLSReadout",
    "algebra_transport",
    "dictionary_update",
    "ridge_objective",
    "ridge_solve",
    "nearest_assignments",
    "partition_parameters",
    "recursive_least_squares",
    "solve_algebra",
    "streaming_rls",
    "variable_projection_objective",
]
