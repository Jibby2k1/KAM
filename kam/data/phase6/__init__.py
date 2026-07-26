"""Deterministic fixtures for every Phase 6 task lane."""

from .dynamics import DynamicsConfig, controlled_prototype, lorenz63, rossler, switching_mackey_glass, switching_narma
from .fixtures import make_dynamics_batch, make_retrieval_batch, make_symbolic_batch
from .language import language_batches, load_text_tokens
from .retrieval import associative_recall, bounded_dyck, mqar, variable_copy
from .symbolic import controlled_symbolic_regimes, regime_purity
from .stream_quality import prequential_evaluate, scheduled_stream

__all__ = [
    "DynamicsConfig",
    "associative_recall",
    "bounded_dyck",
    "controlled_prototype",
    "controlled_symbolic_regimes",
    "language_batches",
    "load_text_tokens",
    "lorenz63",
    "make_dynamics_batch",
    "make_retrieval_batch",
    "make_symbolic_batch",
    "mqar",
    "prequential_evaluate",
    "regime_purity",
    "rossler",
    "scheduled_stream",
    "switching_mackey_glass",
    "switching_narma",
    "variable_copy",
]
