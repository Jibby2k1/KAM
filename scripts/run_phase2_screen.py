from __future__ import annotations

import argparse
from pathlib import Path

from kam.run_suite import load_config, run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase II mechanism screen.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase2/dynamic_screen.yaml"))
    args = parser.parse_args()
    print(f"Completed {len(run_suite(load_config(args.config), config_path=args.config))} screen runs.")


if __name__ == "__main__":
    main()
