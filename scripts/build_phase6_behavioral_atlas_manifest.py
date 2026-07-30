#!/usr/bin/env python3
"""Build a Phase 6.2 bounded L4 profile or complete Stage 0 manifest."""

import argparse
import json

from kam.phase6.behavioral_atlas_manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("l4_profile", "l4_profile_r2", "l4_profile_r3", "stage0"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(write_manifest(args.output, args.stage), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
