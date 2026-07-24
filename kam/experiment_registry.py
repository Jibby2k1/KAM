from __future__ import annotations

import json
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Any


class ExperimentRegistry:
    """SQLite manifest for resumable Phase II suites."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                suite TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                seed INTEGER,
                started_at REAL,
                finished_at REAL,
                output_dir TEXT,
                metrics_json TEXT,
                traceback TEXT
            )"""
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def is_complete(self, run_id: str) -> bool:
        row = self.connection.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return bool(row and row[0] == "complete")

    def start(self, run_id: str, suite: str, config: dict[str, Any], seed: int | None, output_dir: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO runs(run_id,suite,status,config_json,seed,started_at,output_dir) VALUES(?,?,?,?,?,?,?)",
            (run_id, suite, "running", json.dumps(config, sort_keys=True), seed, time.time(), output_dir),
        )
        self.connection.commit()

    def complete(self, run_id: str, metrics: dict[str, Any]) -> None:
        self.connection.execute(
            "UPDATE runs SET status = ?, finished_at = ?, metrics_json = ? WHERE run_id = ?",
            ("complete", time.time(), json.dumps(metrics, sort_keys=True), run_id),
        )
        self.connection.commit()

    def fail(self, run_id: str, error: BaseException) -> None:
        self.connection.execute(
            "UPDATE runs SET status = ?, finished_at = ?, traceback = ? WHERE run_id = ?",
            ("failed", time.time(), "".join(traceback.format_exception(error)), run_id),
        )
        self.connection.commit()
