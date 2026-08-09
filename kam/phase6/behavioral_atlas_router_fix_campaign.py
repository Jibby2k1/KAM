"""Gated replay and pilot campaign for permutation-stable support-identity routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from kam.phase6.behavioral_atlas_permutation_audit import (
    BF16_KL_TOLERANCE,
    BF16_TOP1_TOLERANCE,
    SEMANTIC_TOLERANCE,
    _atomic_json,
)

CAMPAIGN_VERSION = "stage1_support_id_router_validity_campaign_v1"
PILOT_TOKENS = 10_000_000
PILOT_SEEDS_PER_ARM = 3
PILOT_CHECKPOINTS = [0, 1_000_000, 2_000_000, 5_000_000, 10_000_000]


def _id(kind: str, source: str) -> str:
    value = f"{CAMPAIGN_VERSION}:{kind}:{source}"
    return f"p6atlas_{kind}_" + hashlib.sha256(value.encode()).hexdigest()[:16]


def _write_jsonl(rows: list[dict[str, Any]], output: str | Path) -> dict[str, Any]:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    return {
        "rows": len(rows),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "path": str(destination),
    }


def write_replay_manifest(audit_manifest: str | Path, output: str | Path) -> dict[str, Any]:
    audit_rows = [json.loads(line) for line in Path(audit_manifest).read_text().splitlines() if line.strip()]
    if len(audit_rows) != 26:
        raise ValueError(f"router replay requires the locked 26-row audit sample, found {len(audit_rows)}")
    rows = [{
        "row_id": _id("support_replay", str(row["source_row_id"])),
        "source_row_id": str(row["source_row_id"]),
        "arm": row["arm"],
        "seed": row["seed"],
        "device_class": "gpu",
        "replicate": 0,
        "router_tie_breaking": "support_id",
        "compare_legacy_router": True,
        "full_only": True,
        "inferential": False,
        "retraining": False,
    } for row in audit_rows]
    summary = _write_jsonl(rows, output)
    return {**summary, "version": CAMPAIGN_VERSION, "arms": sorted({row["arm"] for row in rows})}


def aggregate_replay(manifest: str | Path, output_root: str | Path, report_root: str | Path) -> dict[str, Any]:
    specs = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    rows = [json.loads(path.read_text()) for path in sorted((Path(output_root) / "rows").glob("*.json"))]
    expected = {row["row_id"] for row in specs}
    passed = [row for row in rows if row.get("status") == "pass"]
    complete = {row["row_id"] for row in passed} == expected
    semantic_failures = [row["source_row_id"] for row in passed if not row["full_matched_permutation"]["semantic"]["passed"]]
    operational_failures = [row["source_row_id"] for row in passed if not row["full_matched_permutation"]["operational"]["passed"]]
    drift_top1 = [float(row["legacy_router_drift"]["operational"]["top1_flip_rate"]) for row in passed]
    drift_kl = [float(row["legacy_router_drift"]["operational"]["predictive_kl"]) for row in passed]
    drift_guard = bool(drift_top1) and max(drift_top1) <= BF16_TOP1_TOLERANCE and max(drift_kl) <= BF16_KL_TOLERANCE
    checks = {
        "all_26_rows_complete": complete and len(specs) == 26,
        "strict_semantic_identity_26_of_26": not semantic_failures and len(passed) == 26,
        "bf16_operational_identity_26_of_26": not operational_failures and len(passed) == 26,
        "legacy_drift_within_registered_operational_tolerances": drift_guard,
        "anchor_identity_preserved": bool(passed) and all(row["source_anchor_sha256"] == row["replication_anchor_sha256"] for row in passed),
    }
    decision = "ROUTER_FIX_REPLAY_PASS" if all(checks.values()) else "ROUTER_FIX_REPLAY_BLOCKED"
    summary = {
        "version": CAMPAIGN_VERSION,
        "decision": decision,
        "checks": checks,
        "expected_rows": len(specs),
        "observed_pass_rows": len(passed),
        "semantic_failures": semantic_failures,
        "operational_failures": operational_failures,
        "legacy_drift": {
            "top1_max": max(drift_top1, default=math.inf),
            "top1_median": float(np.median(drift_top1)) if drift_top1 else math.inf,
            "predictive_kl_max": max(drift_kl, default=math.inf),
            "predictive_kl_median": float(np.median(drift_kl)) if drift_kl else math.inf,
        },
    }
    _atomic_json(Path(output_root) / "router_fix_replay_summary.json", summary)
    report = Path(report_root)
    report.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Support-identity router 26-checkpoint replay",
        "",
        f"**Decision:** `{decision}`",
        "",
        *[f"- {name}: `{value}`" for name, value in checks.items()],
        "",
        f"- Legacy drift maximum top-1 flip rate: {summary['legacy_drift']['top1_max']:.6g}",
        f"- Legacy drift maximum predictive KL: {summary['legacy_drift']['predictive_kl_max']:.6g}",
        "",
        "This post-hoc checkpoint replay is a validity gate only and does not revise original Stage 1 inference.",
    ]
    (report / "ROUTER_FIX_REPLAY.md").write_text("\n".join(lines) + "\n")
    return summary


def write_pilot_manifest(stage1_manifest: str | Path, output: str | Path) -> dict[str, Any]:
    source = [json.loads(line) for line in Path(stage1_manifest).read_text().splitlines() if line.strip()]
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        by_arm[str(row["arm"])].append(row)
    if len(by_arm) != 8:
        raise ValueError(f"pilot requires all 8 Stage 1 arms, found {len(by_arm)}")
    shared = set.intersection(*[{int(row["seed"]) for row in rows} for rows in by_arm.values()])
    seeds = sorted(shared)[:PILOT_SEEDS_PER_ARM]
    if len(seeds) != PILOT_SEEDS_PER_ARM:
        raise ValueError("pilot requires three seeds shared by all arms")
    rows = []
    for arm in sorted(by_arm):
        lookup = {int(row["seed"]): row for row in by_arm[arm]}
        for seed in seeds:
            row = lookup[seed].copy()
            source_id = str(row["row_id"])
            row.update({
                "stage": "stage1_support_id_router_pilot_r1",
                "inferential": False,
                "scientific_role": "prospective_router_validity_pilot",
                "router_tie_breaking": "support_id",
                "supersedes_row_id": source_id,
                "target_tokens": PILOT_TOKENS,
                "validation_token_checkpoints": PILOT_CHECKPOINTS,
                "save_snapshots": True,
            })
            row["row_id"] = _id("support_pilot", source_id)
            rows.append(row)
    summary = _write_jsonl(rows, output)
    return {**summary, "version": CAMPAIGN_VERSION, "arms": sorted(by_arm), "seeds": seeds, "target_tokens": PILOT_TOKENS}


def aggregate_pilot(
    manifest: str | Path, output_root: str | Path, original_root: str | Path, report_root: str | Path
) -> dict[str, Any]:
    specs = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    rows = [json.loads(path.read_text()) for path in sorted((Path(output_root) / "rows" / "behavioral_atlas_v2").glob("*.json"))]
    expected = {row["row_id"] for row in specs}
    passed = [row for row in rows if row.get("status") == "pass"]
    originals = {
        path.stem: json.loads(path.read_text())
        for path in (Path(original_root) / "rows" / "behavioral_atlas_v2").glob("*.json")
    }
    throughput_ratios = [
        float(row["tokens_per_second"]) / float(originals[row["supersedes_row_id"]]["tokens_per_second"])
        for row in passed if row["supersedes_row_id"] in originals
    ]
    original_validation_at_pilot = {
        source_id: next(
            (float(point["validation_loss"]) for point in source["traces"] if int(point["tokens"]) == PILOT_TOKENS),
            math.nan,
        )
        for source_id, source in originals.items()
    }
    loss_changes = [
        float(row["validation_loss"]) / original_validation_at_pilot[row["supersedes_row_id"]] - 1.0
        for row in passed
        if row["supersedes_row_id"] in original_validation_at_pilot
        and math.isfinite(original_validation_at_pilot[row["supersedes_row_id"]])
    ]
    checks = {
        "all_24_rows_complete": len(specs) == 24 and {row["row_id"] for row in passed} == expected,
        "strict_semantic_identity_24_of_24": len(passed) == 24 and all(row["matched_key_expert_permutation"]["passed"] for row in passed),
        "bf16_operational_identity_24_of_24": len(passed) == 24 and all(row["matched_key_expert_permutation"]["operational_within_expected_precision_tolerance"] for row in passed),
        "restart_hashes_24_of_24": len(passed) == 24 and all(row.get("restart_state_hash_match") for row in passed),
        "median_throughput_at_least_80_percent_of_original": len(throughput_ratios) == 24 and float(np.median(throughput_ratios)) >= 0.8,
        "no_absolute_10m_validation_loss_change_over_5_percent": len(loss_changes) == 24 and max(map(abs, loss_changes), default=math.inf) <= 0.05,
    }
    decision = "ROUTER_FIX_PILOT_PASS" if all(checks.values()) else "ROUTER_FIX_PILOT_BLOCKED"
    summary = {
        "version": CAMPAIGN_VERSION,
        "decision": decision,
        "checks": checks,
        "expected_rows": len(specs),
        "observed_pass_rows": len(passed),
        "median_throughput_ratio": float(np.median(throughput_ratios)) if throughput_ratios else 0.0,
        "validation_loss_change_min": min(loss_changes, default=math.inf),
        "validation_loss_change_max": max(loss_changes, default=math.inf),
    }
    _atomic_json(Path(output_root) / "router_fix_pilot_summary.json", summary)
    report = Path(report_root)
    report.mkdir(parents=True, exist_ok=True)
    lines = ["# Support-identity router 24-row pilot", "", f"**Decision:** `{decision}`", "", *[f"- {name}: `{value}`" for name, value in checks.items()], "", "This prospective pilot is noninferential and cannot replace the original Stage 1 campaign."]
    (report / "ROUTER_FIX_PILOT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay_manifest = commands.add_parser("replay-manifest")
    replay_manifest.add_argument("--audit-manifest", required=True)
    replay_manifest.add_argument("--output", required=True)
    replay_final = commands.add_parser("replay-aggregate")
    replay_final.add_argument("--manifest", required=True)
    replay_final.add_argument("--output-root", required=True)
    replay_final.add_argument("--report-root", required=True)
    pilot_manifest = commands.add_parser("pilot-manifest")
    pilot_manifest.add_argument("--stage1-manifest", required=True)
    pilot_manifest.add_argument("--output", required=True)
    pilot_final = commands.add_parser("pilot-aggregate")
    pilot_final.add_argument("--manifest", required=True)
    pilot_final.add_argument("--output-root", required=True)
    pilot_final.add_argument("--original-root", required=True)
    pilot_final.add_argument("--report-root", required=True)
    args = parser.parse_args()
    if args.command == "replay-manifest":
        result = write_replay_manifest(args.audit_manifest, args.output)
    elif args.command == "replay-aggregate":
        result = aggregate_replay(args.manifest, args.output_root, args.report_root)
    elif args.command == "pilot-manifest":
        result = write_pilot_manifest(args.stage1_manifest, args.output)
    else:
        result = aggregate_pilot(args.manifest, args.output_root, args.original_root, args.report_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("decision", "PASS").endswith("BLOCKED"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
