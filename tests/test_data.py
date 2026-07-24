import numpy as np

from kam.data import CopyLanguageDataset, RegimeGrammarDataset, generate_mackey_glass


def test_copy_dataset_shapes_and_mask() -> None:
    dataset = CopyLanguageDataset(size=4, payload_length=5, alphabet_size=6)
    example = dataset[0]
    assert example["inputs"].shape == (12,)
    assert example["targets"].shape == (12,)
    assert int(example["loss_mask"].sum()) == 5


def test_regime_grammar_is_deterministic() -> None:
    dataset = RegimeGrammarDataset(size=4, sequence_length=20, seed=9)
    first = dataset[2]
    second = dataset[2]
    assert np.array_equal(first["inputs"].numpy(), second["inputs"].numpy())


def test_mackey_glass_is_finite_and_reproducible() -> None:
    first = generate_mackey_glass(200, seed=3)
    second = generate_mackey_glass(200, seed=3)
    assert first.shape == (200,)
    assert np.isfinite(first).all()
    assert np.allclose(first, second)
