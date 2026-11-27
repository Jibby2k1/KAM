#!/usr/bin/env python3
"""Compare optimized dot and radial learning under shared elapsed-time limits."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import torch

from benchmark_kam_radial_shape_graphs import train
from run_kam_bitemporal_decoder import make_decoder
from run_kam_curriculum_capture import runtime
from run_kam_curriculum_wallclock import dispatch
from run_kam_radial_curriculum_order import CONFIG, state_sha
from run_kam_radial_dispatch import assess

STEPS = 38400
LIMITS = {"NVIDIA L4": (15, 30, 45, 60), "NVIDIA B200": (6, 12, 18, 24)}


def run(seed, rotation):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    device = torch.cuda.get_device_name()
    budgets = LIMITS[device]
    banks, held, order, rates = runtime(seed, STEPS)
    by_depth = {d: batch * (2*d + 1) for d, batch, _ in CONFIG}
    prefix = [0]
    for depth in order:
        prefix.append(prefix[-1] + by_depth[depth])
    names = ("dot", "radial")
    choices = (names, names[::-1])[rotation % 2]
    result = {}
    for geometry in choices:
        spec = f"{geometry}_without_position"
        model, reference = (make_decoder(spec, seed, "cuda") for _ in range(2))
        initial = make_decoder(spec, seed, "cpu")
        recorded, milestone = {}, set(budgets)

        def kernel(net, length):
            return dispatch(net, f"{geometry}_selective", length)

        def sample(step, setup, updates, loss):
            wall = setup + updates
            reached = sorted(t for t in milestone if wall >= t)
            if reached or step in (9600, 19200):
                reference.load_state_dict(model.state_dict())
                info = {"step": step, "tokens": prefix[step], "depth_updates": dict(Counter(order[:step])),
                        "seconds_with_setup": wall, "update_seconds": updates, "loss": float(loss),
                        "holdout": assess(reference, "eager_fp32", held)}
                for budget in reached:
                    recorded[f"time/{budget}"] = info
                    milestone.remove(budget)
                if step in (9600, 19200):
                    recorded[f"step/{step}"] = info
                print(geometry, step, round(wall, 3), reached, flush=True)
            return not milestone and step >= 19200

        _, _, stats = train(model, {"weights": initial.state_dict()}, banks, order, "shape_graph",
                            rates=rates, kernel_selector=kernel, checkpoint_callback=sample)
        assert not milestone and "step/19200" in recorded
        result[geometry] = {"initial_shared_sha256": state_sha(initial, common=True),
                            "checkpoints": recorded, "setup_seconds": stats["setup_seconds"]}

    assert len({value["initial_shared_sha256"] for value in result.values()}) == 1
    return {"seed": seed, "device": device, "budgets_seconds": budgets,
            "poll_every_updates": 48, "banks": {d: r[-1] for d, r in banks.items()},
            "order_sha256": sha256(json.dumps(order).encode()).hexdigest(),
            "rates_sha256": sha256(rates.cpu().numpy().tobytes()).hexdigest(), "models": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--order", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.seed, a.order), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
