import torch
from kam.factory import make_model


def test_memory_output_suffixes_build_distinct_readouts() -> None:
    for variant in ["DD-v", "DD-a", "DD-b", "DR-v", "DR-a", "DR-b", "RR-v", "RR-a", "RR-b"]:
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
        })
        assert model(torch.randn(2, 16, 2)).shape == (2, 1)
