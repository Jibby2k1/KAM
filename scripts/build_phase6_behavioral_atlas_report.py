#!/usr/bin/env python3
"""Build Stage 0 audit, forecast, figures, and report."""

import argparse
import json

from kam.phase6.behavioral_atlas_analysis import analyze_behavioral_atlas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_behavioral_atlas(args.run_root, args.report_root, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
