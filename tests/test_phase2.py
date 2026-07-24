import torch

from kam.adaptation import NLMSAdapter, prequential_regression
from kam.attention import PairwiseAttentionScore
from kam.data import BoundedDyck2Dataset, MQARDataset, generate_narma
from kam.factory import make_model


def test_expanded_radial_score_matches_direct_reference() -> None:
    torch.manual_seed(4)
    score = PairwiseAttentionScore(2, 6, score_type="radial", radial_metric="diagonal")
    queries = torch.randn(3, 2, 5, 6)
    keys = torch.randn(3, 2, 7, 6)
    direct = score.direct_radial_scores(queries, keys)
    expanded = score.expanded_radial_scores(queries, keys)
    assert torch.allclose(direct, expanded, atol=1e-5, rtol=1e-5)


def test_phase2_variants_have_expected_shapes() -> None:
    for variant in ["D0", "R0", "DD", "DR", "RR"]:
        model = make_model({
            "model_name": variant,
            "task_type": "regression",
            "input_dim": 2,
            "output_dim": 1,
            "d_model": 24,
            "num_heads": 4,
            "num_layers": 1,
            "num_supports": 8,
            "max_seq_len": 16,
            "memory_mode": "both",
        })
        output = model(torch.randn(3, 16, 2))
        assert output.shape == (3, 1)


def test_new_data_generators_are_deterministic_and_finite() -> None:
    mqar = MQARDataset(size=2, sequence_length=32, num_bindings=4, num_queries=2, vocab_size=64, seed=3)
    assert mqar[0]["inputs"].shape == (32,)
    dyck = BoundedDyck2Dataset(size=2, max_depth=8, seed=3)
    assert dyck[0]["targets"].shape == (16,)
    values, inputs = generate_narma(200, order=10, seed=3)
    assert values.shape == inputs.shape == (200,)
    assert torch.isfinite(torch.from_numpy(values)).all()


def test_prequential_nlms_predicts_before_updating() -> None:
    adapter = NLMSAdapter(1, eta=0.5)
    stream = [(torch.tensor([[float(i)]]), torch.tensor([[2.0 * i + 1.0]])) for i in range(1, 5)]
    result = prequential_regression(lambda inputs: inputs, adapter, stream)
    assert result.metrics["samples"] == 4
    assert result.predictions[0].item() != result.targets[0].item()
