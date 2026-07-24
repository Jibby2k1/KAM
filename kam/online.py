from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.no_grad()
def nlms_update(linear: nn.Linear, features: Tensor, targets: Tensor, eta: float = 0.1, eps: float = 1e-6) -> Tensor:
    """Apply a normalized-LMS update to a scalar linear readout.

    The update is averaged over the supplied batch.  It is intended for
    streaming squared-error experiments with a frozen feature extractor.
    """
    if linear.out_features != 1:
        raise ValueError("nlms_update currently supports scalar outputs only.")
    if targets.ndim == 1:
        targets = targets[:, None]
    predictions = linear(features)
    errors = targets - predictions
    augmented = torch.cat(
        [features, torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)],
        dim=-1,
    )
    denominator = eps + augmented.square().sum(dim=-1, keepdim=True)
    delta = eta * (errors / denominator) * augmented
    mean_delta = delta.mean(dim=0)
    linear.weight.add_(mean_delta[:-1][None, :])
    if linear.bias is not None:
        linear.bias.add_(mean_delta[-1:])
    return errors.detach()
