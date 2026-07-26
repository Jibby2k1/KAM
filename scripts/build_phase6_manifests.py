#!/usr/bin/env python3
"""Build a selected Phase 6 profile/full manifest set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kam.phase6.manifest import build_stage_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--mode", choices=("profile", "full"), default="profile")
    args = parser.parse_args()
    stages = args.stage or [f"stage{index}_{name}" for index, name in ((1, "mechanism"), (2, "transformer_comparison"), (3, "router_scaling"), (4, "online_adaptation"), (5, "long_training"), (6, "confirmation"))]
    for stage in stages:
        config = Path("configs/phase6") / f"{stage}.yaml"
        rows = build_stage_manifest(config, mode=args.mode)
        print(f"{stage}: {len(rows)} rows")


if __name__ == "__main__":
    main()
