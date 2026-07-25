from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {k: json_ready(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)


def atomic_torch_save(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Parallel Lustre checkpoint writes can transiently fail with a short
    # PytorchStreamWriter record. Retry the same atomic operation a few times;
    # the experiment remains failed if the filesystem cannot recover.
    for attempt in range(4):
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{attempt}")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
            return
        except (OSError, RuntimeError):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
