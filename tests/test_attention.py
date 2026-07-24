import math

import torch

from kam.attention import KernelSelfAttention


def test_causal_attention_is_normalized_and_masked() -> None:
    torch.manual_seed(0)
    layer = KernelSelfAttention(d_model=16, num_heads=4, score_type="rbf", window=4)
    inputs = torch.randn(2, 7, 16)
    output, weights = layer(inputs, return_weights=True)
    assert output.shape == inputs.shape
    assert weights is not None
    assert weights.shape == (2, 4, 7, 7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)
    for query in range(7):
        for key in range(7):
            allowed = 0 <= query - key < 4
            if not allowed:
                assert float(weights[:, :, query, key].detach().abs().max()) < 1e-6


def test_normalized_rbf_and_dot_weights_are_equivalent_at_matching_temperature() -> None:
    torch.manual_seed(1)
    dimension = 8
    queries = torch.nn.functional.normalize(torch.randn(3, dimension), dim=-1)
    keys = torch.nn.functional.normalize(torch.randn(5, dimension), dim=-1)
    dot_scores = queries @ keys.T / math.sqrt(dimension)
    sigma_squared = math.sqrt(dimension)
    distance_squared = (
        queries.square().sum(dim=-1, keepdim=True)
        + keys.square().sum(dim=-1)[None, :]
        - 2.0 * (queries @ keys.T)
    )
    rbf_scores = -0.5 * distance_squared / sigma_squared
    assert torch.allclose(
        torch.softmax(dot_scores, dim=-1),
        torch.softmax(rbf_scores, dim=-1),
        atol=1e-6,
    )
