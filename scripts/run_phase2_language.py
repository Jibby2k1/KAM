from __future__ import annotations

import argparse
from pathlib import Path

from kam.run_suite import load_config, run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a conditional Phase II language suite.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if not bool(config.get("enabled", False)):
        raise SystemExit("Language stage is disabled until the dynamic-memory gate passes.")
    print(f"Completed {len(run_suite(config, config_path=args.config))} language runs.")


if __name__ == "__main__":
    main()
