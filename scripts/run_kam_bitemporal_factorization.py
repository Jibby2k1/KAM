#!/usr/bin/env python3
"""Audit exact grouped selection and profile pair routing across depth and hardware."""

import argparse
import json
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import torch

from run_kam_bitemporal_noise import bounded_pairs, clock_filter, features, fit, generate


def factored(a, f, left_entity, right_entity, join_bias, *, entities=64):
    """Find the first FP32 winning pair using two linear passes.

    Each alias can upper-bound its compatible scores by that group's best
    fact and its unconstrained scores by the global best fact. A nonnegative
    join bonus makes this bound attainable. Evaluate the chosen alias's
    whole fact row to retain the exhaustive FP32 tie convention.
    """
    by_entity = torch.full((len(a), entities), -torch.inf, dtype=f.dtype, device=f.device)
    by_entity.scatter_reduce_(1, right_entity, f, reduce="amax")
    best_join = (a + by_entity.gather(1, left_entity)) + join_bias
    best_any = a + f.amax(1, keepdim=True)
    alias = torch.maximum(best_join, best_any).argmax(1)
    joined = left_entity.gather(1, alias[:, None]) == right_entity
    final = (a.gather(1, alias[:, None]) + f) + join_bias * joined
    return alias * f.shape[1] + final.argmax(1)


@torch.inference_mode()
def parity_control(device):
    rng = torch.Generator(device=device).manual_seed(21095)
    tested = 0
    for depth in (3, 15, 32, 129):
        for distribution in ("ties", "near_ties", "normal"):
            shape = (257, depth)
            av, fv = (torch.randn(shape, device=device, generator=rng) for _ in range(2))
            if distribution == "ties":
                av, fv = av.round(), fv.round()
            if distribution == "near_ties":
                av, fv = 1024 + av * 0.0001, 1024 + fv * 0.0001
            la, rf = (torch.randint(64, shape, device=device, generator=rng) for _ in range(2))
            for offset in (0.0, 0.02, 2.0, 8.0):
                reference = (av[..., None] + fv[:, None, :] + offset * (la[..., None] == rf[:, None, :])).flatten(1).argmax(1)
                exact = factored(av, fv, la, rf, offset)
                assert torch.equal(reference, exact), (depth, distribution, offset)
                tested += len(reference)
    return tested


def timed(function, repeats, calls):
    for _ in range(8):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_bytes = torch.cuda.memory_allocated()
    walls, kernels = [], []
    for _ in range(repeats):
        t0, t1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize()
        before = perf_counter()
        t0.record()
        for _ in range(calls):
            function()
        t1.record()
        torch.cuda.synchronize()
        walls.append((perf_counter() - before) * 1000 / calls)
        kernels.append(t0.elapsed_time(t1) / calls)
    return {"wall_ms": median(walls), "cuda_ms": median(kernels),
            "incremental_peak_bytes": torch.cuda.max_memory_allocated() - start_bytes,
            "wall_repeats_ms": walls, "gpu_repeats_ms": kernels}


@torch.inference_mode()
def profile(a, model):
    r = Random(56137)
    records = []
    for n in (32, 64, 128, 256, 512, 1024):
        for b in (1, 64):
            q, aliases, facts, _, _, _ = generate(a.seed + 20611, b, n, "mixed", "cuda")
            af, ff, _ = features(q, aliases, facts, include_join=False)
            av, fv = model.local(af, ff)
            ae, fe = aliases[..., 2], facts[..., 1]
            methods = {
                "exhaustive": lambda: bounded_pairs(model, av, fv, ae[..., None] == fe[:, None, :], n),
                "grouped": lambda: factored(av, fv, ae, fe, model.join_bias),
            }
            expected = methods["exhaustive"]()
            assert torch.equal(expected, methods["grouped"]())
            order = r.sample(list(methods), len(methods))
            measurements = {k: timed(methods[k], 5, 100) for k in order}
            records.append({"depth_per_bank": n, "batch": b, "aligned_selections": b,
                            "measurement_order": order, "methods": measurements})
    return records


@torch.inference_mode()
def holdout(a, model):
    records = []
    for n in (20, 32, 64, 128, 256):
        q, aliases, facts, a_gold, f_gold, _ = generate(a.seed + 91_237, a.test_examples, n, "mixed", "cuda")
        exact, masked, invalid = 0, 0, 0
        for start in range(0, len(q), a.batch):
            s = slice(start, start + a.batch)
            af, ff, join = features(q[s], aliases[s], facts[s])
            x, y = model.local(af, ff)
            ae, fe = aliases[s, :, 2], facts[s, :, 1]
            got = factored(x, y, ae, fe, model.join_bias)
            assert torch.equal(got, bounded_pairs(model, x, y, join, 32))
            a_selected, f_selected = got // n, got % n
            a_valid = af[..., 0].bool() & af[..., 1].bool() & clock_filter(aliases[s], q[s], "both")
            f_valid = ff[..., 0].bool() & clock_filter(facts[s], q[s], "both")
            valid = a_valid[..., None] & f_valid[:, None, :] & join
            decision = model.pair(x, y, join).masked_fill(~valid, -torch.inf).flatten(1).argmax(1)
            target = a_gold[s] * n + f_gold[s]
            exact += int(got.eq(target).sum())
            masked += int(decision.eq(target).sum())
            invalid += int((~valid.flatten(1).gather(1, got[:, None])).sum())
            assert bool((a_selected < n).all() and (f_selected < n).all())
        records.append({"depth_per_bank": n, "n": len(q), "unguarded_joint": exact,
                        "hard_masked_joint": masked, "unguarded_invalid": invalid})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=295167)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--train-examples", type=int, default=8192)
    p.add_argument("--test-examples", type=int, default=2048)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    model, loss = fit(a, "one_pass_pair", device)
    bias = float(model.join_bias.detach())
    assert bias >= 0
    parity = parity_control(device)
    results = {"seed": a.seed, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
               "join_bias": bias, "training_loss": loss, "stress_parity_pairs": parity,
               "holdout": holdout(a, model), "profile": profile(a, model)}
    a.output.write_text(json.dumps({**results, "elapsed_seconds": perf_counter() - start}, indent=2,
                                   allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
