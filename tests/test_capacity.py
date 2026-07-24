import torch

from kam.capacity import active_parameter_count, capacity_summary, padding_parameter_count
from kam.factory import make_model


def _spec(target=None):
    return {
        "model_name": "D0", "task_type": "regression", "d_model": 16,
        "num_heads": 4, "num_layers": 1, "num_supports": 8, "max_seq_len": 8,
        "input_dim": 2, "output_dim": 1, "route_features": "projected",
        "route_projection_dim": 8, "parameter_match_target": target,
    }


def test_padding_is_not_active_capacity_and_forward_uses_active_weights():
    model = make_model(_spec(target=6000))
    assert padding_parameter_count(model) > 0
    assert active_parameter_count(model) < sum(parameter.numel() for parameter in model.parameters())
    inputs = torch.randn(2, 8, 2)
    before = model(inputs).detach()
    with torch.no_grad():
        next(parameter for name, parameter in model.named_parameters() if name == "input_layer.weight").add_(0.1)
    after = model(inputs).detach()
    assert not torch.equal(before, after)
    assert capacity_summary(model, _spec(), 8)["padding_parameter_count"] > 0
