#!/usr/bin/env python3
"""Measure per-shape CUDA Graphs on a shuffled mixed-length update schedule."""

import argparse
from hashlib import sha256
import json
from math import pi
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from benchmark_kam_radial_backward import objective, reset
from benchmark_kam_radial_precision import variants
from benchmark_kam_radial_sdpa import predict
from benchmark_kam_radial_training_graph import LR, differences, reset_captured_state
from run_kam_bitemporal_decoder import encode, make_decoder, pack
from run_kam_radial_dispatch import DISPATCH, FUSED, SCHEDULE, assess, choose, panels

DEPTHS = tuple(d for d, _, _ in SCHEDULE)
MODES = ("eager", "shape_graph")


def sequence(cycles, seed):
    order = [d for d in DEPTHS for _ in range(cycles)]
    Random(seed + 765124).shuffle(order)
    assert len(set(order[:20])) == len(DEPTHS) and any(a == b for a, b in zip(order, order[1:]))
    return order


def train(model, checkpoint, banks, order, mode, *, rates=None, kernel_selector=None, checkpoint_callback=None):
    reset(model, checkpoint)
    torch.cuda.empty_cache()
    if rates is None:
        phases = torch.linspace(0, pi, len(order), device="cuda")
        rates = LR * (0.2 + 0.8 * (1 + phases.cos()) / 2)
    assert len(rates) == len(order)
    learning_rate = rates[0].clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True, capturable=True)
    indexes = {d: data[2] for d, data in banks.items()}
    global_step = torch.zeros(1, dtype=torch.long, device="cuda")
    cursors = {d: torch.zeros_like(global_step) for d in indexes}
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    setup_start = perf_counter()

    def step(depth, index, rate):
        scene, perm, _, _ = banks[depth]
        tokens, key, payload = encode(pack(scene, index), perm[index], hide_query_clock=True)
        kernel = choose(model, DISPATCH, depth) if kernel_selector is None else kernel_selector(model, depth)
        learning_rate.copy_(rate)
        optimizer.zero_grad(set_to_none=True)
        cost = objective(predict(model, kernel, tokens), key, payload)
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        return cost.detach()

    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for k in range(3):
            for depth in indexes:
                step(depth, indexes[depth][k], rates[k])
    torch.cuda.current_stream().wait_stream(warm)
    graphs, output = {}, {}
    if mode == "shape_graph":
        for depth in indexes:
            graph = torch.cuda.CUDAGraph()
            optimizer.zero_grad(set_to_none=True)
            cursor = cursors[depth]
            with torch.cuda.graph(graph):
                index = indexes[depth].index_select(0, cursor)[0]
                rate = rates.index_select(0, global_step)[0]
                output[depth] = step(depth, index, rate)
                cursor.add_(1)
                global_step.add_(1)
            graphs[depth] = graph
    reset_captured_state(model, optimizer, checkpoint)
    global_step.zero_()
    for cursor in cursors.values():
        cursor.zero_()
    torch.cuda.synchronize()
    setup = perf_counter() - setup_start

    hardware_start = torch.cuda.Event(enable_timing=True)
    hardware_end = torch.cuda.Event(enable_timing=True)
    by_length = {d: 0 for d in indexes}
    started = perf_counter()
    hardware_start.record()
    executed = len(order)
    if mode == "shape_graph":
        excluded = 0.
        for j, depth in enumerate(order, 1):
            graphs[depth].replay()
            if checkpoint_callback is not None and j % 48 == 0:
                torch.cuda.synchronize()
                paused = perf_counter()
                if checkpoint_callback(j, setup, paused - started - excluded, output[depth]):
                    executed = j
                    break
                torch.cuda.synchronize()
                excluded += perf_counter() - paused
        last = output[order[executed - 1]]
    else:
        for i, depth in enumerate(order):
            last = step(depth, indexes[depth][by_length[depth]], rates[i])
            by_length[depth] += 1
    hardware_end.record()
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    if mode == "shape_graph":
        elapsed -= excluded
    memory = {"peak_extra_allocated_bytes": torch.cuda.max_memory_allocated() - baseline,
              "peak_extra_reserved_bytes": torch.cuda.max_memory_reserved() - reserved}
    if mode == "shape_graph":
        assert int(global_step[0]) == executed
        assert all(int(cursors[d][0]) == order[:executed].count(d) for d in indexes)
    assert torch.isfinite(last)
    assert all(int(v["step"]) == executed for v in optimizer.state.values())
    params = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
    moments = {name: torch.cat([s[name].flatten().cpu() for s in optimizer.state.values()])
               for name in ("exp_avg", "exp_avg_sq")}
    config = {"setup_seconds": setup, "wall_seconds": elapsed, "stream_seconds": hardware_start.elapsed_time(hardware_end)/1000,
              "loss": float(last), **memory}
    return params, moments, config


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    assert sha256(a.checkpoint.read_bytes()).hexdigest() == a.checkpoint_sha256
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["arm"] == "radial_without_position" and checkpoint["hide_query_clock"]
    model = variants(checkpoint)[FUSED]
    reference = make_decoder(checkpoint["arm"], checkpoint["seed"], "cuda")
    orders = sequence(a.cycles, a.seed)
    banks, holdout = panels(a.seed, len(orders))
    found, states, optimizers = {}, {}, {}
    names = MODES if a.order % 2 == 0 else tuple(reversed(MODES))
    for name in (*names, *(f"{key}_repeat" for key in reversed(names))):
        weights, moments, stats = train(model, checkpoint, banks, orders, name.removesuffix("_repeat"))
        reference.load_state_dict(weights)
        stats["heldout"] = assess(reference, "eager_fp32", holdout)
        states[name], optimizers[name], found[name] = weights, moments, stats
        print(name, stats["wall_seconds"], flush=True)
    comparisons = {name: {"weights": differences(states["eager"], weights),
                          "moments": differences(optimizers["eager"], optimizers[name]),
                          "final_loss_equal": found[name]["loss"] == found["eager"]["loss"],
                          "heldout_equal": found[name]["heldout"] == found["eager"]["heldout"]}
                   for name, weights in states.items()}
    return {"device": torch.cuda.get_device_name(), "seed": a.seed, "cycles": a.cycles,
            "checkpoint_sha256": a.checkpoint_sha256, "schedule_sha256": sha256(json.dumps(orders).encode()).hexdigest(),
            "banks": {d: banks[d][-1] for d in DEPTHS}, "runs": found, "comparisons": comparisons}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--order", type=int, required=True)
    p.add_argument("--cycles", type=int, default=256)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
