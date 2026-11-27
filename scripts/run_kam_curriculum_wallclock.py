#!/usr/bin/env python3
"""Compare dot and radial curricula under a common measured L4 update-time budget."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter
from types import MethodType

import torch
from torch.nn import functional as F

from benchmark_kam_radial_backward import fixture, objective
from benchmark_kam_radial_sdpa import labels, predict
from kam.attention import KernelSelfAttention
from run_kam_bitemporal_decoder import encode, make_decoder, pack
from run_kam_radial_curriculum_order import CONFIG, POLICIES, setup, state_sha, training_order
from run_kam_radial_dispatch import DISPATCH, assess, choose

MODES = ("dot_eager", "dot_selective", "radial_selective")
DOT_FLASH = "dot_flash_bf16"
EAGER = "eager_fp32"
LIMITS = (25., 45., 60.)
BLOCK = 120


def flash_dot(attention, inputs, return_weights=False):
    assert not return_weights and attention.window is None and attention.relative_bias is None
    assert attention.score.score_type == "dot" and attention.dropout.p == 0
    batch, length, _ = inputs.shape
    qkv = attention.qkv(inputs).view(batch, length, 3, attention.num_heads, attention.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attention.score._prepare(q, k)
    result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    result = result.transpose(1, 2).contiguous().view(batch, length, attention.d_model)
    return attention.output(result), None


def dispatch(model, mode, depth):
    if mode == "dot_eager":
        return EAGER
    if mode == "radial_selective":
        return choose(model, DISPATCH, depth)
    fused = mode == "dot_selective" and depth == 512
    for block in model.blocks:
        block.context.forward = MethodType(flash_dot if fused else KernelSelfAttention.forward, block.context)
    return DOT_FLASH if fused else EAGER


def probe(seed):
    x, stamp = fixture(seed + 35631, 16, 512)
    inputs, entity, value = x
    outputs = {}
    for mode in (EAGER, "eager_bf16", DOT_FLASH):
        net = make_decoder("dot_without_position", seed, "cuda")
        if mode == DOT_FLASH:
            dispatch(net, "dot_selective", 512)
        result = predict(net, mode, inputs)
        loss = objective(result, entity, value)
        loss.backward()
        vector = torch.cat([v.grad.detach().float().flatten() for v in net.parameters() if v.grad is not None])
        assert torch.isfinite(vector).all()
        outputs[mode] = (float(loss.detach()), result.detach().float(), vector)
    checks = {}
    for baseline in (EAGER, "eager_bf16"):
        a, b = outputs[baseline], outputs[DOT_FLASH]
        checks[baseline] = {"abs_logit": float((b[1] - a[1]).abs().max()),
                            "decision_matches": int(labels(b[1]).eq(labels(a[1])).all(1).sum()),
                            "gradient_relative_l2": float((b[2] - a[2]).norm() / a[2].norm().clamp_min(1e-12)),
                            "loss_delta": b[0] - a[0]}
    return {"fixture_sha256": stamp, "checks": checks}


def order_for(seed):
    prefix = training_order(9600, POLICIES[1], seed)
    rng = Random(seed + 50113)
    order = prefix.copy()
    for _ in range(3):
        tail = prefix.copy()
        rng.shuffle(tail)
        order.extend(tail)
    return order


def train(seed, mode, order, banks, held):
    architecture = "dot" if mode.startswith("dot") else "radial"
    arm = f"{architecture}_without_position"
    model = make_decoder(arm, seed, "cuda")
    reference = make_decoder(arm, seed, "cuda")
    state = state_sha(model, common=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, fused=True)
    depths = {d: 0 for d, _, _ in CONFIG}
    model.train()
    for depth, _, _ in CONFIG:
        records, permutation, batches, _ = banks[depth]
        indices = batches[0]
        warmup = encode(pack(records, indices), permutation[indices], hide_query_clock=True)
        objective(predict(model, dispatch(model, mode, depth), warmup[0]), *warmup[1:]).backward()
        model.zero_grad(set_to_none=True)
    assert state == state_sha(model, common=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    begin = perf_counter()
    elapsed = 0.
    tokens, milestones = 0, {}
    tracked = set(LIMITS)
    last_is_long, path = None, None
    for step, depth in enumerate(order, 1):
        records, permutation, batches, _ = banks[depth]
        selected = batches[depths[depth]]
        depths[depth] += 1
        encoded, good_entity, good_value = encode(pack(records, selected), permutation[selected],
                                                 hide_query_clock=True)
        is_long = depth == 512
        if is_long != last_is_long:
            path = dispatch(model, mode, depth)
            last_is_long = is_long
        optimizer.param_groups[0]["lr"] = 0.0004 * min(step / 64, 1.0)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(predict(model, path, encoded), good_entity, good_value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += encoded.shape[0] * encoded.shape[1]
        if step % BLOCK != 0:
            continue
        torch.cuda.synchronize()
        elapsed += perf_counter() - begin
        reached = {b for b in tracked if elapsed >= b}
        if reached or step in (4800, 9600):
            assert torch.isfinite(loss)
            memory = torch.cuda.max_memory_allocated() - baseline
            reference.load_state_dict(model.state_dict())
            outcome = assess(reference, EAGER, held)
            info = {"seconds": elapsed, "step": step, "tokens": tokens, "depth_updates": depths.copy(),
                    "holdout": outcome, "peak_extra_allocated_bytes": memory}
            for b in sorted(reached):
                milestones[str(b)] = info
                tracked.remove(b)
            if step in (4800, 9600):
                milestones[f"step/{step}"] = info
            print(mode, step, round(elapsed, 3), sorted(reached), flush=True)
        if not tracked:
            break
        torch.cuda.synchronize()
        begin = perf_counter()
    assert not tracked
    return {"initial_shared_sha256": state, "checkpoints": milestones}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--order", required=True, type=int, choices=range(3))
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    order = order_for(a.seed)
    assert len(order) == 38400 and all(order.count(d) == 9600 for d, _, _ in CONFIG)
    banks, held = setup(a.seed, len(order))
    audited = probe(a.seed)
    results = {}
    for mode in (*MODES[a.order:], *MODES[:a.order]):
        results[mode] = train(a.seed, mode, order, banks, held)
    assert len({m["initial_shared_sha256"] for m in results.values()}) == 1
    result = {"device": torch.cuda.get_device_name(), "seed": a.seed, "shape": CONFIG, "budgets": LIMITS,
              "order_sha256": sha256(" ".join(map(str, order)).encode()).hexdigest(),
              "banks": {str(i): v[-1] for i, v in banks.items()}, "dot_precision_probe": audited, "models": results}
    a.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
