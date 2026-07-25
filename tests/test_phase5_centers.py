import torch
from torch.utils.data import DataLoader, TensorDataset

from kam.factory import make_model
from kam.run_suite import _initialize_data_centers


def _model(num_layers: int = 1):
    torch.manual_seed(3)
    return make_model({
        "model_name": "KC-LV",
        "task_type": "regression",
        "input_dim": 2,
        "output_dim": 1,
        "d_model": 16,
        "num_heads": 4,
        "num_layers": num_layers,
        "num_supports": 4,
        "max_seq_len": 6,
        "memory_output": "both",
    })


def test_random_normal_center_initialization_is_a_sampler_noop() -> None:
    model = _model()
    original = model.blocks[0].memory.memory_keys.detach().clone()

    class LoaderThatMustNotBeRead:
        def __iter__(self):
            raise AssertionError("random_normal must return before reading data")

    saved = _initialize_data_centers(model, LoaderThatMustNotBeRead(), "random_normal")
    assert torch.equal(model.blocks[0].memory.memory_keys, original)
    assert torch.equal(saved["block_0"], original)


def test_sampled_centers_use_projected_block_local_key_coordinates() -> None:
    model = _model()
    inputs = torch.arange(48, dtype=torch.float32).reshape(4, 6, 2) / 48.0
    loader = DataLoader(TensorDataset(inputs, torch.zeros(4, 1)), batch_size=4)

    with torch.no_grad():
        hidden = model.input_layer(inputs)
        hidden += model._position_encoding(6, hidden.device, hidden.dtype)
        block = model.blocks[0]
        context_update, _ = block.context(
            block.context_norm(hidden), return_weights=False
        )
        memory_input = block.memory_norm(hidden + context_update)
        projected = block.memory.project_keys(memory_input)
        pool = projected.permute(0, 2, 1, 3).reshape(
            -1, block.memory.num_heads, block.memory.head_dim
        )
        selected = torch.linspace(0, pool.shape[0] - 1, 4).long()
        expected = pool[selected].permute(1, 0, 2)

    _initialize_data_centers(model, loader, "sampled_training_points")
    assert torch.allclose(model.blocks[0].memory.memory_keys, expected)
    assert not model.blocks[0].memory.memory_keys.requires_grad


def test_each_layer_receives_its_own_center_state() -> None:
    model = _model(num_layers=2)
    inputs = torch.randn(8, 6, 2)
    loader = DataLoader(TensorDataset(inputs, torch.zeros(8, 1)), batch_size=4)
    saved = _initialize_data_centers(model, loader, "sampled_training_points")
    assert set(saved) == {"block_0", "block_1"}
    assert not torch.allclose(saved["block_0"], saved["block_1"])
