from __future__ import annotations

import json
from pathlib import Path

from kam.phase6.behavioral_atlas_manifest import build_behavioral_atlas_rows
from kam.phase6.behavioral_atlas_router_fix_campaign import write_pilot_manifest, write_replay_manifest


def test_router_fix_replay_manifest_keeps_locked_26_rows(tmp_path: Path) -> None:
    source = tmp_path / "audit.jsonl"
    source.write_text("".join(json.dumps({"source_row_id": f"row_{i}", "arm": f"arm_{i % 8}", "seed": i}) + "\n" for i in range(26)))
    output = tmp_path / "replay.jsonl"
    summary = write_replay_manifest(source, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["rows"] == 26
    assert len({row["row_id"] for row in rows}) == 26
    assert all(row["router_tie_breaking"] == "support_id" and row["full_only"] for row in rows)
    assert all(row["compare_legacy_router"] and not row["inferential"] and not row["retraining"] for row in rows)


def test_router_fix_pilot_is_eight_arms_by_three_shared_seeds(tmp_path: Path) -> None:
    source = tmp_path / "stage1.jsonl"
    source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in build_behavioral_atlas_rows("stage1_core_lifecycle")))
    output = tmp_path / "pilot.jsonl"
    summary = write_pilot_manifest(source, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["rows"] == 24
    assert len(summary["arms"]) == 8
    assert len(summary["seeds"]) == 3
    assert all(row["target_tokens"] == 10_000_000 for row in rows)
    assert all(row["router_tie_breaking"] == "support_id" and not row["inferential"] for row in rows)
