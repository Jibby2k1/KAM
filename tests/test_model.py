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
    )
    inputs = torch.randn(5, 16, 2)
    features, _ = model.regression_features(inputs)
    assert features.shape == (5, 32 + 4 * 10)
    predictions = model(inputs)
    assert predictions.shape == (5, 1)
