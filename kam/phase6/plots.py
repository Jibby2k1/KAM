from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return str(destination)


def plot_learning_curves(history: dict[str, Iterable[float]], path: str | Path) -> str:
    fig, axis = plt.subplots(figsize=(6, 4))
    for label, values in history.items():
        axis.plot(list(values), label=label)
    axis.set(xlabel="step", ylabel="loss", title="Learning curves")
    axis.legend()
    return _save(fig, path)


def plot_prediction_true_error(y_true: Iterable[float], prediction: Iterable[float], path: str | Path, *, log_error: bool = True) -> str:
    truth = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(prediction), dtype=float)
    error = pred - truth
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(truth, label="true")
    axes[0].plot(pred, label="prediction", alpha=0.8)
    axes[0].legend()
    axes[0].set_ylabel("value")
    axes[1].plot(error, color="tab:red")
    axes[1].set_ylabel("error")
    axes[2].plot(np.abs(error) + 1e-12, color="tab:purple")
    if log_error:
        axes[2].set_yscale("log")
    axes[2].set(xlabel="sample", ylabel="|error|")
    return _save(fig, path)


def plot_memory_diagnostics(metrics: dict[str, Iterable[float]], path: str | Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fields = [("routing_entropy", "routing entropy"), ("effective_support_count", "effective supports"), ("dead_support_fraction", "dead fraction"), ("load_balance_error", "load balance error")]
    for axis, (field, title) in zip(axes.flat, fields):
        if field in metrics:
            axis.plot(list(metrics[field]))
        axis.set_title(title)
        axis.set_xlabel("step")
    return _save(fig, path)


def plot_router_load(loads: Iterable[float], path: str | Path) -> str:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(list(loads), bins=20)
    axis.set(xlabel="tokens per support", ylabel="supports", title="Router load")
    return _save(fig, path)


__all__ = ["plot_learning_curves", "plot_memory_diagnostics", "plot_prediction_true_error", "plot_router_load"]
