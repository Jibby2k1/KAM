import torch

from kam.model import KAMSequenceModel


def test_language_model_shapes() -> None:
    model = KAMSequenceModel(
        task="language",
        vocab_size=13,
        d_model=32,
        num_heads=4,
        num_layers=2,
        num_supports=12,
        max_seq_len=32,
    )
    tokens = torch.randint(0, 13, (3, 20))
    logits, diagnostics = model(tokens, return_weights=True)
    assert logits.shape == (3, 20, 13)
    assert len(diagnostics.context_weights) == 2
    assert len(diagnostics.memory_weights) == 2
    assert diagnostics.memory_weights[-1].shape == (3, 4, 20, 12)


def test_regression_features_include_memory_routing() -> None:
    model = KAMSequenceModel(
        task="regression",
        input_dim=2,
        output_dim=1,
        d_model=32,
        num_heads=4,
        num_layers=1,
        num_supports=10,
        max_seq_len=16,
        expose_memory_weights=True,
        append_routes_to_readout=True,
    )
    inputs = torch.randn(5, 16, 2)
    features, _ = model.regression_features(inputs)
    assert features.shape == (5, 32 + 4 * 10)
    predictions = model(inputs)
    assert predictions.shape == (5, 1)


def test_diagnostics_do_not_append_routes_to_values_only_readout() -> None:
    model = KAMSequenceModel(
        task="regression",
        input_dim=2,
        output_dim=1,
        d_model=32,
        num_heads=4,
        num_layers=1,
        num_supports=10,
        max_seq_len=16,
        memory_output="residual",
        return_attention_for_diagnostics=True,
        append_routes_to_readout=False,
        apply_memory_residual=True,
    )
    features, diagnostics = model.regression_features(
        torch.randn(5, 16, 2), return_weights=True
    )
    assert features.shape == (5, 32)
    assert diagnostics.memory_weights
    assert model.route_feature_dim == 0
