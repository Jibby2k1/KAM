#!/usr/bin/env python3
"""Fork identical optimizer states to test late learning-rate decay."""

import argparse
from copy import deepcopy
import json
from math import cos, pi
from pathlib import Path
from time import perf_counter

import torch

from benchmark_kam_radial_backward import objective
from benchmark_kam_radial_sdpa import predict
from run_kam_bitemporal_decoder import encode, make_decoder, pack
from run_kam_curriculum_wallclock import EAGER, dispatch, order_for
from run_kam_radial_curriculum_order import CONFIG, setup, state_sha
from run_kam_radial_dispatch import assess

OH = 9600
END = 19200
RATE = 0.0004
CHECKPOINTS = (OH, 12000, 14400, 16800, END)


def trajectory(model, reference, optimizer, order, banks, holdout, seen, begin_step, last_step, decay):
    checkpoints, elapsed = {}, 0.
    numeric = None
    model.train()
    torch.cuda.synchronize()
    timer = perf_counter()
    for step in range(begin_step + 1, last_step + 1):
        depth = order[step - 1]
        records, permutation, batches, _ = banks[depth]
        selection = batches[seen[depth]]
        seen[depth] += 1
        x, key, fact = encode(pack(records, selection), permutation[selection], hide_query_clock=True)
        long = depth == 512
        if numeric is None or numeric[0] != long:
            numeric = long, dispatch(model, model.training_path, depth)
        scale = min(step / 64, 1)
        if decay and step > OH:
            scale *= .1 + .9 * (1 + cos(pi * (step - OH) / (END - OH))) * .5
        optimizer.param_groups[0]["lr"] = RATE * scale
        optimizer.zero_grad(set_to_none=True)
        loss = objective(predict(model, numeric[1], x), key, fact)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in CHECKPOINTS:
            torch.cuda.synchronize()
            elapsed += perf_counter() - timer
            assert torch.isfinite(loss)
            reference.load_state_dict(model.state_dict())
            checkpoints[str(step)] = {"elapsed_update_seconds": elapsed, "depth_updates": seen.copy(),
                                      "loss": float(loss.detach()), "rate": RATE * scale,
                                      "held_out": assess(reference, EAGER, holdout)}
            print(model.training_path, decay, step, round(elapsed, 3), flush=True)
            torch.cuda.synchronize()
            timer = perf_counter()
    return checkpoints


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    banks, holdout = setup(a.seed, 38400)
    order = order_for(a.seed)[:END]
    output = {}
    for j, geom in enumerate(("dot", "radial") if a.order % 2 == 0 else ("radial", "dot")):
        arm = f"{geom}_without_position"
        net, ref = (make_decoder(arm, a.seed, "cuda") for _ in range(2))
        optimizer = torch.optim.AdamW(net.parameters(), lr=RATE, fused=True)
        net.training_path = f"{geom}_selective"
        seen = {d: 0 for d, _, _ in CONFIG}
        prefix = trajectory(net, ref, optimizer, order, banks, holdout, seen, 0, OH, False)
        weights = deepcopy(net.state_dict())
        state = deepcopy(optimizer.state_dict())
        fixed = state_sha(net)
        paths = {}
        for decay in ((True, False) if (a.order + j) % 2 else (False, True)):
            net.load_state_dict(weights)
            assert state_sha(net) == fixed
            optimizer.load_state_dict(deepcopy(state))
            paths["cosine" if decay else "constant"] = trajectory(net, ref, optimizer, order, banks, holdout,
                                                                  seen.copy(), OH, END, decay)
        assert all(paths["cosine"][p]["depth_updates"] == paths["constant"][p]["depth_updates"] for p in paths["constant"])
        output[geom] = {"prefix": prefix, "fork_weights_sha256": fixed, "branches": paths}
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "bank_sha256": {d: v[-1] for d, v in banks.items()},
            "learning_rate": RATE, "checkpoint_updates": CHECKPOINTS, "models": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--order", required=True, type=int, choices=(0, 1, 2))
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
