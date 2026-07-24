"""Small JSONL/CSV table helpers for Phase IV artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def write_table(path: str | Path, rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    path = Path(path)
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in materialized:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        temporary.replace(path)
        return {"format": "jsonl", "path": str(path)}
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return {"format": "csv", "path": str(path)}


def read_table(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
