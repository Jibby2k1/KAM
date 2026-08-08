#!/usr/bin/env python3
"""Build a Phase 6.1 parameter-dynamics pilot or main manifest."""

import argparse
import json
from kam.phase6.parameter_dynamics_manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(write_manifest(args.output, args.stage), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
