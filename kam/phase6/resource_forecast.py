from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .resource import ResourceForecast, forecast_transformer_memory


def forecast_row(row: dict[str, Any]) -> dict[str, Any]:
    d_model = int(row.get("d_model", 64))
    layers = int(row.get("n_layers", 2))
    d_ff = int(row.get("d_ff", 4 * d_model))
    forecast = forecast_transformer_memory(
        d_model=d_model,
        n_layers=layers,
        d_ff=d_ff,
        num_supports=int(row.get("num_supports", 0)),
        top_k=int(row.get("top_k", 0)),
        expert_mode=str(row.get("expert_mode", "vector")),
    )
    return asdict(forecast)


__all__ = ["ResourceForecast", "forecast_row", "forecast_transformer_memory"]
