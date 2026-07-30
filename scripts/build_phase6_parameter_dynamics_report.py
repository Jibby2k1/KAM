#!/usr/bin/env python3
import argparse, json
from kam.phase6.parameter_dynamics_analysis import analyze_parameter_dynamics

parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True); parser.add_argument("--report-root", required=True); parser.add_argument("--manifest", required=True)
args = parser.parse_args(); print(json.dumps(analyze_parameter_dynamics(args.run_root, args.report_root, args.manifest), indent=2, sort_keys=True))
