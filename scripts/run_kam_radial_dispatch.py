#!/usr/bin/env python3
"""Compare eager, fused, and depth-selective updates on a capable checkpoint."""

import argparse
from copy import deepcopy
from functools import partial
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter
from types import MethodType

import torch

from benchmark_kam_radial_backward import objective, reset
from benchmark_kam_radial_precision import stable_forward, variants
from benchmark_kam_radial_sdpa import labels, predict
from kam.attention import KernelSelfAttention
from run_kam_bitemporal_decoder import encode, fingerprint, make_decoder, pack, record_order
from run_kam_bitemporal_noise import generate

FUSED = "balanced_split_flash_bf16"
DISPATCH = "fp32_short_flash_long"
MODES = ("eager_fp32", "eager_bf16", "augmented_flash_bf16", FUSED, DISPATCH)
SCHEDULE = ((20, 100, 4096), (128, 16, 2048), (512, 4, 1024))


def choose(model, mode, depth):
    if mode != DISPATCH:
        return mode
    fused = depth == 512
    candidate = partial(stable_forward, balanced=True, split_bias=True) if fused else KernelSelfAttention.forward
    for block in model.blocks:
        block.context.forward = MethodType(candidate, block.context)
    return FUSED if fused else "eager_fp32"


def panels(seed, updates):
    banks, held = {}, {}
    for depth, batch, count in SCHEDULE:
        scene = generate(seed + 104801 + depth, count, depth, "mixed", "cuda")
        order = record_order(count, depth, seed + 164019 + depth, "cuda")
        rng = torch.Generator().manual_seed(seed + 402179 + depth)
        batches = torch.randint(count, (updates // len(SCHEDULE), batch), generator=rng)
        stamp = fingerprint(scene, order), sha256(batches.numpy().tobytes()).hexdigest()
        banks[depth] = scene, order, batches.to("cuda"), stamp
        samples = generate(seed + 579071 + depth, 512, depth, "mixed", "cuda")
        sample_order = record_order(512, depth, seed + 641921 + depth, "cuda")
        held[depth] = samples, sample_order, fingerprint(samples, sample_order)
    return banks, held


def fit(model, mode, banks, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, fused=True)
    model.train()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    tokens, losses = 0, []
    for step in range(1, steps + 1):
        depth, _, _ = SCHEDULE[(step - 1) % len(SCHEDULE)]
        scene, order, indices, _ = banks[depth]
        selection = indices[(step - 1) // len(SCHEDULE)]
        x, entity, fact = encode(pack(scene, selection), order[selection], hide_query_clock=True)
        update = choose(model, mode, depth)
        optimizer.param_groups[0]["lr"] = 0.0004 * min(step / 30, 1.0) * (
            0.3 + 0.7 * (1 + math.cos(math.pi * step / steps)) * 0.5)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(predict(model, update, x), entity, fact)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += x.shape[0] * x.shape[1]
        if step % 300 == 0 or step == steps:
            assert torch.isfinite(loss)
            losses.append({"step": step, "loss": float(loss.detach()), "tokens": tokens})
    torch.cuda.synchronize()
    return {"steps": steps, "tokens": tokens, "seconds": perf_counter() - started,
            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated() - resident, "losses": losses}


@torch.inference_mode()
def assess(model, mode, held):
    output = {}
    model.eval()
    for depth, (scene, order, stamp) in held.items():
        method = choose(model, mode, depth)
        matches = []
        for i in range(0, len(order), 16):
            sl = slice(i, i + 16)
            x, entity, value = encode(pack(scene, sl), order[sl], hide_query_clock=True)
            decoded = labels(predict(model, method, x))
            matches.extend(decoded.eq(torch.stack((entity, value), -1)).tolist())
        output[str(depth)] = {"fixture_sha256": stamp, "outcomes": matches}
    return output


def run(a):
    torch.set_float32_matmul_precision("highest")
    assert a.steps % len(SCHEDULE) == 0 and a.steps >= 300
    digest = sha256(a.checkpoint.read_bytes()).hexdigest()
    assert digest == a.checkpoint_sha256
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["arm"] == "radial_without_position" and checkpoint["hide_query_clock"]
    source = variants(checkpoint)
    modes = {name: source[name] for name in MODES if name != DISPATCH}
    modes[DISPATCH] = deepcopy(source[FUSED])
    del source
    baseline = make_decoder(checkpoint["arm"], checkpoint["seed"], "cuda")
    reset(baseline, checkpoint)
    banks, held = panels(a.seed, a.steps)
    initial = assess(baseline, "eager_fp32", held)
    outputs = {}
    rotate = a.order % len(MODES)
    for mode in (*MODES[rotate:], *MODES[:rotate]):
        model = modes[mode]
        reset(model, checkpoint)
        profile = fit(model, mode, banks, a.steps)
        baseline.load_state_dict(model.state_dict())
        outputs[mode] = {"training": profile, "common_fp32": assess(baseline, "eager_fp32", held),
                         "native": assess(model, mode, held)}
        model.zero_grad(set_to_none=True)
        print(mode, profile["seconds"], flush=True)
    return {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "seed": a.seed,
            "checkpoint_sha256": digest, "checkpoint_tokens": checkpoint["train_tokens"],
            "banks": {str(k): v[-1] for k, v in banks.items()}, "schedule": SCHEDULE, "order": a.order,
            "initial": initial, "modes": outputs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--order", required=True, type=int)
    p.add_argument("--steps", default=1200, type=int)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
