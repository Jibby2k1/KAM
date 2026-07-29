#!/usr/bin/env python3
"""Write and summarize the immutable confirmation-v2 manifest."""

from __future__ import annotations

import argparse
import json

from kam.phase6.confirmation_manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/phase6/confirmation_v2/manifest.jsonl")
    args = parser.parse_args()
    print(json.dumps(write_manifest(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
