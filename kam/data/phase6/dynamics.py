from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DynamicsConfig:
    length: int = 256
    observation_noise: float = 0.0
    process_noise: float = 0.0
    seed: int = 0


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def controlled_prototype(length: int = 256, regime: int = 0, separation: float = 1.0, seed: int = 0) -> tuple[Tensor, Tensor]:
    generator = _generator(seed)
    x = torch.randn(length, 2, generator=generator) * 0.1
    y = torch.zeros(length, 1)
    for step in range(1, length):
        previous = x[step - 1]
        rotation = 0.6 + 0.15 * regime
        x[step, 0] = torch.tanh(rotation * previous[0] - 0.2 * previous[1] + separation * 0.02)
        x[step, 1] = torch.tanh(0.2 * previous[0] + rotation * previous[1] - separation * 0.01)
        y[step] = x[step, 0]
    return x, y


def switching_mackey_glass(length: int = 512, delays: tuple[int, ...] = (17, 23), switch_points: tuple[int, ...] = (256,), config: DynamicsConfig | None = None) -> tuple[Tensor, Tensor, Tensor]:
    config = config or DynamicsConfig(length=length)
    generator = _generator(config.seed)
    signal = torch.zeros(length + max(delays) + 2)
    signal[: max(delays) + 1] = 1.2 + 0.05 * torch.randn(max(delays) + 1, generator=generator)
    regimes = torch.zeros(length, dtype=torch.long)
    for step in range(length):
        regime = sum(step >= point for point in switch_points) % len(delays)
        regimes[step] = regime
        lag = delays[regime]
        current = signal[step + max(delays)]
        delayed = signal[step + max(delays) - lag]
        noise = config.process_noise * torch.randn((), generator=generator)
        signal[step + max(delays) + 1] = current + 0.2 * (0.1 * delayed / (1 + delayed**10) - 0.1 * current) + noise
    observed = signal[max(delays) + 1 : max(delays) + 1 + length].unsqueeze(-1)
    observation_noise = torch.randn(observed.shape, generator=generator, dtype=observed.dtype, device=observed.device)
    observed = observed + config.observation_noise * observation_noise
    return observed[:-1], observed[1:], regimes


def switching_narma(length: int = 512, regimes: int = 2, config: DynamicsConfig | None = None) -> tuple[Tensor, Tensor, Tensor]:
    config = config or DynamicsConfig(length=length)
    generator = _generator(config.seed)
    inputs = torch.rand(length, generator=generator)
    output = torch.zeros(length)
    labels = torch.arange(length) % regimes
    for step in range(2, length):
        coefficient = 0.2 + 0.08 * labels[step]
        output[step] = coefficient * output[step - 1] + 0.05 * output[step - 1] * output[step - 2] + 1.5 * inputs[step - 2] * inputs[step - 1] + 0.1
    return torch.stack((inputs, output), -1)[:-1], output[1:].unsqueeze(-1), labels[:-1]


def lorenz63(length: int = 512, dt: float = 0.01, seed: int = 0) -> tuple[Tensor, Tensor]:
    state = torch.tensor([1.0, 1.0, 1.0])
    trajectory = torch.zeros(length, 3)
    for step in range(length):
        trajectory[step] = state
        x, y, z = state
        state = state + dt * torch.tensor((10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z))
    return trajectory[:-1], trajectory[1:]


def rossler(length: int = 512, dt: float = 0.02) -> tuple[Tensor, Tensor]:
    state = torch.tensor([0.1, 0.0, 0.0])
    trajectory = torch.zeros(length, 3)
    for step in range(length):
        trajectory[step] = state
        x, y, z = state
        state = state + dt * torch.tensor((-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)))
    return trajectory[:-1], trajectory[1:]


__all__ = ["DynamicsConfig", "controlled_prototype", "lorenz63", "rossler", "switching_mackey_glass", "switching_narma"]
