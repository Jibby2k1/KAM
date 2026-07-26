#!/usr/bin/env python3
"""Rebuild all final Phase 6 overnight reports and machine-readable exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kam.phase6.overnight_analysis import final_aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results/phase6/overnight")
    parser.add_argument("--report-root", default="reports/phase6/overnight")
    args = parser.parse_args()
    result = final_aggregate(Path(args.run_root), Path(args.report_root))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
