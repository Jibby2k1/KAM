from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _encode(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _decode(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    if text[0] in "[{" and text[-1] in "]}":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    fields = sorted({key for row in materialized for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _encode(row.get(key)) for key in fields})
    temporary.replace(path)


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [{key: _decode(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_table(path: str | Path, rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Write a table, using Parquet when pyarrow is present and CSV/JSONL otherwise."""
    path = Path(path)
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(materialized)
            temporary = path.with_suffix(path.suffix + ".tmp")
            pq.write_table(table, temporary)
            temporary.replace(path)
            return {"format": "parquet", "path": str(path)}
        except ImportError:
            csv_path = path.with_suffix(".csv")
            jsonl_path = path.with_suffix(".jsonl")
            write_csv(csv_path, materialized)
            write_jsonl(jsonl_path, materialized)
            note = path.with_suffix(path.suffix + ".unavailable.json")
            write_json(note, {"requested": str(path), "fallback_csv": str(csv_path), "fallback_jsonl": str(jsonl_path), "reason": "pyarrow is not installed"})
            return {"format": "csv+jsonl", "path": str(csv_path)}
    if path.suffix == ".jsonl":
        write_jsonl(path, materialized)
        return {"format": "jsonl", "path": str(path)}
    write_csv(path, materialized)
    return {"format": "csv", "path": str(path)}


def read_table(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            return pq.read_table(path).to_pylist()
        except (ImportError, FileNotFoundError):
            fallback = path.with_suffix(".jsonl")
            if fallback.exists():
                return read_jsonl(fallback)
            fallback = path.with_suffix(".csv")
            if fallback.exists():
                return read_csv(fallback)
            raise
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    return read_csv(path)
