#!/usr/bin/env python3
"""Extract exact registered rows for an infrastructure-only timeout repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def exact_subset(
    manifest: Path,
    indices: list[int],
    *,
    expected_sha256: str | None = None,
) -> tuple[list[dict], dict]:
    payload = manifest.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"manifest SHA-256 mismatch: expected {expected_sha256}, found {digest}")

    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(indices) != len(set(indices)):
        raise ValueError("repair indices must be unique")
    if not indices:
        raise ValueError("at least one repair index is required")
    if min(indices) < 0 or max(indices) >= len(rows):
        raise IndexError(f"repair index outside manifest range 0..{len(rows) - 1}")

    selected = [rows[index] for index in indices]
    if len({row["row_id"] for row in selected}) != len(selected):
        raise ValueError("selected rows do not have unique row IDs")
    audit = {
        "repair_type": "infrastructure_timeout_exact_rerun",
        "scientific_fields_modified": False,
        "source_manifest": str(manifest),
        "source_manifest_sha256": digest,
        "source_row_count": len(rows),
        "source_indices": indices,
        "row_ids": [row["row_id"] for row in selected],
    }
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--indices", required=True, help="comma-separated zero-based manifest indices")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    rows, audit = exact_subset(args.manifest, indices, expected_sha256=args.expected_sha256)
    output = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    audit["repair_manifest"] = str(args.output)
    audit["repair_manifest_sha256"] = hashlib.sha256(output.encode("utf-8")).hexdigest()
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
