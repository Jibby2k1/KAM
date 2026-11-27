#!/usr/bin/env python3
"""Capture a fixed cycle of radial updates and check parameter and optimizer parity."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from benchmark_kam_radial_backward import objective, reset
from benchmark_kam_radial_precision import variants
from benchmark_kam_radial_sdpa import predict
from run_kam_bitemporal_decoder import encode, make_decoder, pack
from run_kam_radial_dispatch import DISPATCH, FUSED, SCHEDULE, assess, choose, panels

POLICIES = ("eager_fp32", DISPATCH)
LR = 0.0002


def cycle(model, policy, optimizer, banks, indices, rates=None):
    losses = []
    for i, ((depth, _, _), selection) in enumerate(zip(SCHEDULE, indices, strict=True)):
        scene, order, _, _ = banks[depth]
        inputs, entity, record = encode(pack(scene, selection), order[selection], hide_query_clock=True)
        kernel = choose(model, policy, depth)
        if rates is not None:
            optimizer.param_groups[0]["lr"].copy_(rates[i])
        optimizer.zero_grad(set_to_none=True)
        loss = objective(predict(model, kernel, inputs), entity, record)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.detach())
    return losses


@torch.no_grad()
def reset_captured_state(model, optimizer, checkpoint):
    # The graph retains references to these allocations. Reset their contents.
    reset(model, checkpoint)
    for state in optimizer.state.values():
        for value in state.values():
            value.zero_()


def train(model, policy, checkpoint, banks, count, method, schedule):
    reset(model, checkpoint)
    torch.cuda.empty_cache()
    rates = None
    lr = LR
    if schedule == "cosine":
        progress = torch.linspace(0, math.pi, 3 * count, device="cuda")
        rates = (LR * (0.2 + 0.8 * (1 + progress.cos()) / 2)).reshape(count, 3)
        lr = rates[0, 0].clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True, capturable=True)
    indices = tuple(banks[d][2] for d, _, _ in SCHEDULE)
    static = [batch[0].clone() for batch in indices]
    static_rates = None if rates is None else rates[0].clone()
    cursor = torch.zeros(1, dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    reserve = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    setup = perf_counter()

    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for j in range(3):
            cycle(model, policy, optimizer, banks, [x[j] for x in indices], None if rates is None else rates[j])
    torch.cuda.current_stream().wait_stream(warm)

    graph, stored_loss = None, None
    if method != "regular":
        graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            if method == "cursor":
                selected = [x.index_select(0, cursor)[0] for x in indices]
                selected_rates = None if rates is None else rates.index_select(0, cursor)[0]
                stored_loss = cycle(model, policy, optimizer, banks, selected, selected_rates)
                cursor.add_(1)
            else:
                stored_loss = cycle(model, policy, optimizer, banks, static, static_rates)
    reset_captured_state(model, optimizer, checkpoint)
    cursor.zero_()
    torch.cuda.synchronize()
    setup = perf_counter() - setup

    started = perf_counter()
    if graph is None:
        for j in range(count):
            losses = cycle(model, policy, optimizer, banks, [x[j] for x in indices], None if rates is None else rates[j])
    else:
        for j in range(count):
            if method == "graph":
                for slot, data in zip(static, indices, strict=True):
                    slot.copy_(data[j])
                if rates is not None:
                    static_rates.copy_(rates[j])
            graph.replay()
        losses = stored_loss
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    memory = {"peak_extra_allocated_bytes": torch.cuda.max_memory_allocated() - baseline,
              "peak_extra_reserved_bytes": torch.cuda.max_memory_reserved() - reserve}
    assert all(torch.isfinite(value) for value in losses)
    if method == "cursor":
        assert int(cursor[0]) == count

    active, inactive = [], []
    for name, parameter in model.named_parameters():
        if parameter in optimizer.state:
            state = optimizer.state[parameter]
            assert int(state["step"]) == 3 * count
            active.append(parameter)
        else:
            inactive.append(name)
    weights = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
    assert active and all(torch.equal(weights[k], checkpoint["weights"][k]) for k in inactive)
    moments = {field: torch.cat([optimizer.state[p][field].flatten().cpu() for p in active])
               for field in ("exp_avg", "exp_avg_sq")}
    stats = {"seconds": elapsed, "setup_seconds": setup, **memory, "final_losses": [float(x) for x in losses],
             "active_optimizer_parameters": len(active), "inactive_parameters": inactive}
    return stats, weights, moments


def differences(reference, alternate):
    xy = torch.cat([x.flatten().float() for x in reference.values()])
    xx = torch.cat([alternate[k].flatten().float() for k in reference])
    delta = (xy - xx).abs()
    return {"equal": torch.equal(xy, xx), "max_absolute": float(delta.max()),
            "relative_l2": float(delta.norm() / xy.norm().clamp_min(1e-20))}


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(a.deterministic)
    assert a.cycles >= 4
    digest = sha256(a.checkpoint.read_bytes()).hexdigest()
    assert digest == a.checkpoint_sha256
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["arm"] == "radial_without_position" and checkpoint["hide_query_clock"]
    models = variants(checkpoint)
    chosen = {"eager_fp32": models["eager_fp32"], DISPATCH: deepcopy(models[FUSED])}
    del models
    reference = make_decoder(checkpoint["arm"], checkpoint["seed"], "cuda")
    banks, held = panels(a.seed, 3 * a.cycles)
    result = {}
    for policy in a.policies:
        weights, moments, values = {}, {}, {}
        controls = ("regular", "graph", "cursor") if a.cursor else ("regular", "graph")
        shift = a.order % len(controls)
        order = controls[shift:] + controls[:shift]
        methods = (*order, *(f"{p}_repeat" for p in reversed(order))) if a.repeat else order
        for method in methods:
            stats, weights[method], moments[method] = train(
                chosen[policy], policy, checkpoint, banks, a.cycles, method.removesuffix("_repeat"), a.learning_rate)
            reference.load_state_dict(weights[method])
            stats["held_out"] = assess(reference, "eager_fp32", held)
            values[method] = stats
            print(policy, method, stats["seconds"], flush=True)
        result[policy] = {
            "runs": values,
            "weight_difference": differences(weights["regular"], weights["graph"]),
            "moment_difference": {k: differences({k: v}, {k: moments["graph"][k]})
                                  for k, v in moments["regular"].items()},
        }
        if a.repeat:
            result[policy]["same_path_repeat"] = {
                p: {
                    "weights": differences(weights[p], weights[f"{p}_repeat"]),
                    "moments": differences(moments[p], moments[f"{p}_repeat"]),
                }
                for p in order
            }
        if a.cursor:
            result[policy]["cursor_difference"] = {
                "weights": differences(weights["regular"], weights["cursor"]),
                "moments": differences(moments["regular"], moments["cursor"]),
            }
    return {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "seed": a.seed,
            "checkpoint_sha256": digest, "schedule": SCHEDULE, "order": a.order, "cycles": a.cycles,
            "deterministic_algorithms": a.deterministic,
            "learning_rate": a.learning_rate,
            "update_tokens": sum((2 * d + 1) * b for d, b, _ in SCHEDULE) * a.cycles,
            "banks": {str(d): value[-1] for d, value in banks.items()}, "policies": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--order", required=True, type=int)
    p.add_argument("--cycles", default=256, type=int)
    p.add_argument("--repeat", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--cursor", action="store_true")
    p.add_argument("--learning-rate", choices=("constant", "cosine"), default="constant")
    p.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
