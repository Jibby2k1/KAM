"""Small, explicit decoder components used by the Phase 6 campaign.

The Phase 6 transformer is intentionally separate from the historical
``KAMSequenceModel``.  This keeps the old experiments reproducible while
giving the new sparse-memory comparisons a common decoder interface.
"""

from .config import TransformerConfig
from .decoder import DecoderBlock, ModernDecoder
from .layers import RMSNorm, SinusoidalPosition, SwiGLU
from .baselines import ArchitectureSpec, architecture_spec, build_baseline, parameter_budget, scale_config

__all__ = [
    "DecoderBlock",
    "ArchitectureSpec",
    "ModernDecoder",
    "RMSNorm",
    "SinusoidalPosition",
    "SwiGLU",
    "TransformerConfig",
    "architecture_spec",
    "build_baseline",
    "parameter_budget",
    "scale_config",
]
