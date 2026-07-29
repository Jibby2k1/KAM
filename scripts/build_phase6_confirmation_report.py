#!/usr/bin/env python3
"""Build the locked Phase 6 confirmation report and exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kam.phase6.confirmation_analysis import final_aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results/phase6/confirmation_v2")
    parser.add_argument("--report-root", default="reports/phase6/confirmation_v2")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    result = final_aggregate(
        Path(args.run_root),
        Path(args.report_root),
        Path(args.manifest) if args.manifest else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
