from __future__ import annotations

import argparse
import copy
from pathlib import Path

from kam.run_suite import load_config, run_suite


def expand_seed_grid(config: dict, seeds: list[int], output_root: Path | None) -> dict:
    template_runs = list(config.get("runs", []))
    if not template_runs:
        raise ValueError("The base config must contain template runs.")
    expanded = []
    for seed in seeds:
        for template in template_runs:
            run = copy.deepcopy(template)
            base_id = str(run["run_id"]).rsplit("_s", 1)[0]
            run["seed"] = int(seed)
            run["run_id"] = f"{base_id}_s{seed}"
            expanded.append(run)
    result = copy.deepcopy(config)
    result["suite_name"] = str(config.get("suite_name", "phase2")) + "_paired_screen"
    result["output_root"] = str(output_root or Path(config.get("output_root", "results/phase2")) / "paired_screen")
    result["study_database"] = str(Path(result["output_root"]) / "study.sqlite")
    result["metrics_table"] = str(Path(result["output_root"]) / "all_metrics.csv")
    result["runs"] = expanded
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired multi-seed Phase II confirmation/screening grid.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    config = expand_seed_grid(load_config(args.config), args.seeds, args.output_root)
    print(f"Expanded to {len(config['runs'])} paired runs across seeds {args.seeds}.")
    print(f"Completed {len(run_suite(config, config_path=args.config))} confirmation runs.")


if __name__ == "__main__":
    main()
