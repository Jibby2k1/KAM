"""Controlled symbolic regime language with explicit factor controls."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ControlledSymbolicRegimeDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, size: int, sequence_length: int = 64, alphabet_size: int = 12,
                 regime_count: int = 4, transition_entropy: float = 0.5,
                 emission_overlap: float = 0.2, return_probability: float = 0.5,
                 explicit_regime_token: bool = False, seed: int = 0) -> None:
        self.size = size
        self.sequence_length = sequence_length
        self.alphabet_size = alphabet_size
        self.regime_count = regime_count
        self.transition_entropy = transition_entropy
        self.emission_overlap = emission_overlap
        self.return_probability = return_probability
        self.explicit_regime_token = explicit_regime_token
        self.seed = seed
        self.bos = alphabet_size
        self.regime_offset = alphabet_size + 1
        self.vocab_size = alphabet_size + 1 + (regime_count if explicit_regime_token else 0)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        regime = int(rng.integers(0, self.regime_count))
        visited = {regime}
        symbols = []
        regimes = []
        previous = int(rng.integers(0, self.alphabet_size))
        current = int(rng.integers(0, self.alphabet_size))
        for position in range(self.sequence_length + 1):
            if position and rng.random() >= self.transition_entropy:
                next_regime = regime
            else:
                returning = sorted(visited - {regime})
                unvisited = sorted(set(range(self.regime_count)) - visited)
                if returning and rng.random() < self.return_probability:
                    # A return is a transition to a previously visited,
                    # different regime—not another dwell in the current one.
                    next_regime = int(rng.choice(returning))
                elif unvisited:
                    next_regime = int(rng.choice(unvisited))
                else:
                    different = [
                        candidate
                        for candidate in range(self.regime_count)
                        if candidate != regime
                    ]
                    next_regime = int(rng.choice(different))
            regime = next_regime
            visited.add(regime)
            center = int((regime * self.alphabet_size) / self.regime_count)
            if rng.random() < self.emission_overlap:
                symbol = int(rng.integers(0, self.alphabet_size))
            else:
                symbol = int((center + current + previous) % self.alphabet_size)
            symbols.append(symbol)
            regimes.append(regime)
            previous, current = current, symbol
        full = np.asarray([self.bos] + symbols, dtype=np.int64)
        inputs = full[:-1]
        targets = full[1:]
        if self.explicit_regime_token:
            inputs = inputs.copy()
            inputs[0] = self.regime_offset + regimes[0]
            # Reveal the newly active regime at every later transition while
            # retaining ordinary symbol inputs during regime dwell periods.
            for position in range(1, len(inputs)):
                if regimes[position] != regimes[position - 1]:
                    inputs[position] = self.regime_offset + regimes[position]
        mask = np.ones_like(targets, dtype=np.float32)
        mask[0] = 0.0
        return {"inputs": torch.from_numpy(inputs), "targets": torch.from_numpy(targets), "loss_mask": torch.from_numpy(mask), "metadata": torch.from_numpy(np.asarray(regimes[:-1], dtype=np.int64))}


def make_symbolic_splits(*, seed: int, train_size: int, validation_size: int, test_size: int, **kwargs):
    common = dict(kwargs)
    return (
        ControlledSymbolicRegimeDataset(train_size, seed=seed, **common),
        ControlledSymbolicRegimeDataset(validation_size, seed=seed + 1000003, **common),
        ControlledSymbolicRegimeDataset(test_size, seed=seed + 2000006, **common),
    )
