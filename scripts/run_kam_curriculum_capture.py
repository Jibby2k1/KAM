#!/usr/bin/env python3
"""Compare whole from-scratch dot and radial curricula with per-shape graphs."""

import argparse
from hashlib import sha256
import json
from math import pi
from pathlib import Path

import torch

from benchmark_kam_radial_shape_graphs import train
from benchmark_kam_radial_training_graph import differences
from run_kam_bitemporal_decoder import make_decoder
from run_kam_curriculum_wallclock import dispatch, order_for
from run_kam_radial_curriculum_order import setup, state_sha
from run_kam_radial_dispatch import assess

STEPS = 19200
FORK = 9600
MODES = ("eager", "shape_graph")


def runtime(seed, steps=STEPS):
    order = order_for(seed)[:steps]
    banks, holdout = setup(seed, steps)
    step = torch.arange(1, steps + 1, dtype=torch.float32, device="cuda")
    warm = (step / 64).clamp(max=1)
    cosine = ((step - FORK) / (STEPS - FORK)).clamp(min=0, max=1) * pi
    return banks, holdout, order, .0004 * warm * (.1 + .9 * (1 + cosine.cos()) / 2)


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    banks, holdout, order, rates = runtime(a.seed)
    output = {}
    geometries = ("dot", "radial") if a.order % 2 == 0 else ("radial", "dot")
    for j, g in enumerate(geometries):
        arm = f"{g}_without_position"
        initial = make_decoder(arm, a.seed, "cpu")
        checkpoint = {"weights": initial.state_dict()}
        shared = state_sha(initial, common=True)
        model, reference = (make_decoder(arm, a.seed, "cuda") for _ in range(2))
        trials, params, moments = {}, {}, {}
        modes = MODES if (a.order + j) % 2 == 0 else tuple(reversed(MODES))

        def kernel(net, depth):
            return dispatch(net, f"{g}_selective", depth)

        for mode in modes:
            params[mode], moments[mode], stats = train(model, checkpoint, banks, order, mode, rates=rates, kernel_selector=kernel)
            reference.load_state_dict(params[mode])
            stats["heldout"] = assess(reference, "eager_fp32", holdout)
            trials[mode] = stats
            print(g, mode, round(stats["wall_seconds"], 3), flush=True)
        eager, graph = [trials[k] for k in MODES]
        output[g] = {"initial_shared_sha256": shared, "runs": trials,
                     "parity": {"weights": differences(*[params[k] for k in MODES]),
                                "moments": differences(*[moments[k] for k in MODES]),
                                "final_loss_equal": eager["loss"] == graph["loss"],
                                "heldout_equal": eager["heldout"] == graph["heldout"]}}
    assert len({r["initial_shared_sha256"] for r in output.values()}) == 1
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "steps": STEPS,
            "banks": {d: r[-1] for d, r in banks.items()}, "order_sha256": sha256(json.dumps(order).encode()).hexdigest(),
            "rates_sha256": sha256(rates.cpu().numpy().tobytes()).hexdigest(), "models": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--order", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
