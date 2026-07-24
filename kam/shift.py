from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.mackey_glass import MackeyGlassDataset, generate_mackey_glass
from .factory import make_model
from .model import KAMSequenceModel
from .online import nlms_update
from .utils import choose_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KAM under a Mackey-Glass parameter shift.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/shift"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--recovery-window", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def predict_features(model: KAMSequenceModel, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features, _ = model.regression_features(inputs, return_weights=False)
    return model.readout(features), features


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("task") != "mackey-glass":
        raise ValueError("Shift evaluation requires a Mackey-Glass checkpoint.")
    static_model = make_model(checkpoint["model_spec"]).to(device)
    static_model.load_state_dict(checkpoint["model_state"])
    if not isinstance(static_model, KAMSequenceModel) or static_model.task != "regression":
        raise ValueError("NLMS shift evaluation requires a KAM regression model.")
    adaptive_model = copy.deepcopy(static_model)
    static_model.eval()
    adaptive_model.eval()
    for parameter in adaptive_model.parameters():
        parameter.requires_grad_(False)

    metadata = checkpoint["data_metadata"]
    mean = float(metadata["mean"])
    std = float(metadata["std"])
    window = int(checkpoint["model_spec"]["max_seq_len"])
    raw = generate_mackey_glass(
        args.length + window,
        tau=args.tau,
        beta=args.beta,
        seed=args.seed,
    )
    standardized = (raw - mean) / std
    dataset = MackeyGlassDataset(standardized, window=window)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rows: list[dict[str, float]] = []
    static_errors: list[float] = []
    adaptive_errors: list[float] = []
    for index, (inputs, target) in enumerate(loader):
        inputs = inputs.to(device)
        target = target.to(device)
        with torch.no_grad():
            static_prediction, _ = predict_features(static_model, inputs)
            adaptive_prediction, features = predict_features(adaptive_model, inputs)
        static_error = float((target - static_prediction).squeeze().cpu())
        adaptive_error = float((target - adaptive_prediction).squeeze().cpu())
        nlms_update(adaptive_model.readout, features, target, eta=args.eta)
        static_errors.append(static_error)
        adaptive_errors.append(adaptive_error)
        rows.append(
            {
                "index": index,
                "target": float(target.squeeze().cpu()),
                "static_prediction": float(static_prediction.squeeze().cpu()),
                "adaptive_prediction": float(adaptive_prediction.squeeze().cpu()),
                "static_error": static_error,
                "adaptive_error": adaptive_error,
            }
        )

    with (args.output / "shift_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    static_sq = np.square(static_errors)
    adaptive_sq = np.square(adaptive_errors)
    recovery = min(args.recovery_window, len(static_sq))
    summary = {
        "checkpoint": str(args.checkpoint),
        "shift_tau": args.tau,
        "shift_beta": args.beta,
        "eta": args.eta,
        "static_mse": float(static_sq.mean()),
        "adaptive_mse": float(adaptive_sq.mean()),
        "static_early_mse": float(static_sq[:recovery].mean()),
        "adaptive_early_mse": float(adaptive_sq[:recovery].mean()),
        "static_late_mse": float(static_sq[-recovery:].mean()),
        "adaptive_late_mse": float(adaptive_sq[-recovery:].mean()),
        "integrated_error_ratio": float(adaptive_sq.sum() / max(static_sq.sum(), 1e-12)),
    }
    save_json(args.output / "shift_metrics.json", summary)

    smoothing = max(1, min(64, len(static_sq) // 20))
    kernel = np.ones(smoothing) / smoothing
    static_smooth = np.convolve(static_sq, kernel, mode="valid")
    adaptive_smooth = np.convolve(adaptive_sq, kernel, mode="valid")
    plt.figure(figsize=(6.2, 3.8))
    plt.plot(static_smooth, label="frozen readout")
    plt.plot(adaptive_smooth, label="NLMS-adapted readout")
    plt.xlabel("shifted-stream step")
    plt.ylabel("smoothed squared error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "shift_error.png", dpi=170)
    plt.close()
    print(summary)


if __name__ == "__main__":
    main()
