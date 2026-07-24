from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def get_nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect KAM metrics.json files into one CSV.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("summary.csv"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.root.rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload.get("final_validation", {})
        rows.append(
            {
                "path": str(path.parent),
                "task": payload.get("task"),
                "model": payload.get("model"),
                "device": payload.get("device"),
                "parameters": payload.get("parameter_count"),
                "mse": validation.get("mse"),
                "mae": validation.get("mae"),
                "cross_entropy": validation.get("cross_entropy"),
                "accuracy": validation.get("accuracy"),
                "perplexity": validation.get("perplexity"),
                "support_regime_purity": get_nested(
                    payload, "diagnostics", "support_regime_purity"
                ),
                "total_seconds": payload.get("total_seconds"),
            }
        )
    if not rows:
        raise SystemExit(f"No metrics.json files found below {args.root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
