import torch

from kam.factory import make_model


def _model(variant: str):
    torch.manual_seed(11)
    return make_model({
        "model_name": variant,
        "task_type": "regression",
        "input_dim": 2,
        "output_dim": 1,
        "d_model": 24,
        "num_heads": 4,
        "num_layers": 1,
        "num_supports": 8,
        "max_seq_len": 12,
        "route_features": "projected",
        "route_projection_dim": 16,
        "return_attention_for_diagnostics": True,
    })


def test_phase5_component_aliases_preserve_distinct_semantics() -> None:
    routes = _model("DD-A")
    values = _model("DD-V")
    both = _model("DD-B")

    assert routes.memory_output == "routes"
    assert routes.append_routes_to_readout
    assert not routes.apply_memory_residual
    assert routes.route_feature_dim == 16

    assert values.memory_output == "residual"
    assert not values.append_routes_to_readout
    assert values.apply_memory_residual
    assert values.route_feature_dim == 0

    assert both.memory_output == "both"
    assert both.append_routes_to_readout
    assert both.apply_memory_residual
    assert both.route_feature_dim == 16


def test_routes_only_and_values_only_have_distinct_forward_dependencies() -> None:
    inputs = torch.randn(3, 12, 2)
    routes = _model("DD-A").eval()
    values = _model("DD-V").eval()

    routes_before = routes(inputs).detach()
    values_before = values(inputs).detach()
    with torch.no_grad():
        routes.blocks[0].memory.memory_values.add_(5.0)
        values.blocks[0].memory.memory_values.add_(5.0)
    assert torch.allclose(routes_before, routes(inputs), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(values_before, values(inputs))


def test_component_gradients_match_semantic_paths() -> None:
    inputs = torch.randn(2, 12, 2)
    for variant, values_should_train, output_should_train in (
        ("DD-A", False, False),
        ("DD-V", True, True),
        ("DD-B", True, True),
    ):
        model = _model(variant)
        model(inputs).square().mean().backward()
        memory = model.blocks[0].memory
        assert (memory.memory_values.grad is not None) is values_should_train
        assert (memory.output.weight.grad is not None) is output_should_train
        assert memory.memory_keys.grad is not None


def test_values_only_can_return_attention_without_readout_aliasing() -> None:
    model = _model("DD-V")
    prediction, diagnostics = model(torch.randn(2, 12, 2), return_weights=True)
    assert prediction.shape == (2, 1)
    assert diagnostics.memory_weights
    assert model.readout.in_features == model.d_model
