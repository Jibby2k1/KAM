import torch
from torch.utils.data import DataLoader

from kam.data import VariableCopyLanguageDataset, variable_copy_collate


def test_variable_copy_uses_variable_lengths_and_padding() -> None:
    dataset = VariableCopyLanguageDataset(size=16, min_payload_length=3, max_payload_length=8, seed=5)
    examples = [dataset[index] for index in range(16)]
    lengths = {int(example["inputs"].shape[0]) for example in examples}
    assert len(lengths) > 1
    batch = next(iter(DataLoader(dataset, batch_size=8, collate_fn=variable_copy_collate)))
    assert batch["inputs"].shape == batch["targets"].shape
    assert torch.isfinite(batch["loss_mask"]).all()
    assert batch["loss_mask"].sum() > 0
