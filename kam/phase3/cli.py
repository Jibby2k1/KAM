from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="kam-phase3", description="Manifest-driven Phase III campaign tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, default=None)

    audit = sub.add_parser("audit")
    audit.add_argument("--root", type=Path, default=Path("."))
    audit.add_argument("--output-root", type=Path, default=Path("results/phase3"))

    run_row = sub.add_parser("run-row")
    run_row.add_argument("--manifest", type=Path, required=True)
    run_row.add_argument("--row-id", type=int, required=True)
    run_row.add_argument("--run-root", type=Path, required=True)
    run_row.add_argument("--device", default="auto")
    run_row.add_argument("--resume", action="store_true")

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--run-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, default=None)
    aggregate.add_argument("--report-root", type=Path, default=Path("reports/phase3"))

    report = sub.add_parser("report")
    report.add_argument("--run-root", type=Path, required=True)
    report.add_argument("--report-root", type=Path, default=Path("reports/phase3"))

    gate = sub.add_parser("gate")
    gate.add_argument("--aggregate", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--gate-config", type=Path, default=Path("configs/phase3/gates.yaml"))
    gate.add_argument("--gate-type", choices=["primary", "audit"], default="primary")

    args = parser.parse_args()
    if args.command == "build-manifest":
        from .manifest import build_manifest
        result = build_manifest(args.config, args.output)
        print(json.dumps({"rows": len(result)}, indent=2))
    elif args.command == "audit":
        from .audit import run_audit
        print(json.dumps(run_audit(args.root, args.output_root), indent=2, sort_keys=True, default=str))
    elif args.command == "run-row":
        from .manifest import load_manifest
        from .run_array import execute_row
        row = next(row for row in load_manifest(args.manifest) if int(row.get("row_id", -1)) == args.row_id)
        print(json.dumps(execute_row(row, args.run_root, device_name=args.device, resume=args.resume), indent=2, sort_keys=True, default=str))
    elif args.command in {"aggregate", "report"}:
        from .aggregate import aggregate
        print(json.dumps(aggregate(args.run_root, getattr(args, "output", None), args.report_root), indent=2, sort_keys=True, default=str))
    elif args.command == "gate":
        from .gate import audit_gate, primary_gate
        result = audit_gate(".", args.output) if args.gate_type == "audit" else primary_gate(args.aggregate, args.output, args.gate_config)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        if not result.get("pass", False):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
