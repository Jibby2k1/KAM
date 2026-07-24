from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .baselines import BudgetedKLMSRegressor
from .data.mackey_glass import generate_mackey_glass
from .utils import save_json, set_seed


def windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.stack([series[index : index + window] for index in range(len(series) - window)])
    targets = series[window:]
    return inputs, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-budget quantized KLMS baseline.")
    parser.add_argument("--output", type=Path, default=Path("outputs/klms"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--series-length", type=int, default=12000)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--tau", type=float, default=17.0)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--novelty", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = generate_mackey_glass(args.series_length, tau=args.tau, seed=args.seed)
    split = int(0.75 * len(raw))
    mean = raw[:split].mean()
    std = raw[:split].std() + 1e-8
    standardized = (raw - mean) / std
    train_x, train_y = windows(standardized[:split], args.window)
    validation_x, validation_y = windows(standardized[split - args.window :], args.window)

    model = BudgetedKLMSRegressor(
        budget=args.budget,
        sigma=args.sigma,
        eta=args.eta,
        novelty=args.novelty,
    )
    train_errors = [model.update(x, float(y)) for x, y in zip(train_x, train_y, strict=True)]
    predictions = np.asarray([model.predict(x) for x in validation_x])
    errors = validation_y - predictions
    summary = {
        "train_mse": float(np.mean(np.square(train_errors))),
        "validation_mse": float(np.mean(np.square(errors))),
        "validation_mae": float(np.mean(np.abs(errors))),
        "supports_used": len(model.centers),
        "budget": args.budget,
        "sigma": args.sigma,
        "eta": args.eta,
        "novelty": args.novelty,
    }
    save_json(args.output / "metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
