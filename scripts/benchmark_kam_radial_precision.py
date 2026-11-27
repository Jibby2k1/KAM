#!/usr/bin/env python3
"""Compare two BF16 encodings of the radial score at fixed FlashAttention width."""

import argparse
from copy import deepcopy
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F

from benchmark_kam_radial_backward import assess, benchmark, fixture, gradient_probe, reset, continue_training
from benchmark_kam_radial_sdpa import make_variants

DEPTHS = (20, 128, 512)
SEEDS = (829114, 829128, 829162)


def stable_forward(attention, inputs, return_weights=False, *, balanced, split_bias, last_only=False):
    assert not return_weights and attention.dropout.p == 0
    assert attention.relative_bias is None and attention.window is None
    score = attention.score
    batch, length, _ = inputs.shape
    head = attention.head_dim
    qkv = attention.qkv(inputs).view(batch, length, 3, attention.num_heads, head)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    if last_only:
        q = q[..., -1:, :]
    fmt = q.dtype
    if balanced:
        compute_type = torch.promote_types(fmt, torch.float32)
        q, k = q.to(compute_type), k.to(compute_type)
    q, k = score._prepare(q, k)
    q, k = score._prepare(q, k)
    metric = score.metric()[None, :, None, :]
    bandwidth = score.bandwidth()[None, :, None, None]
    if balanced:
        scale = metric.sqrt() / bandwidth
        q, k = q * scale, k * scale
        bias = -0.5 * k.square().sum(-1, keepdim=True)
    else:
        inverse_sq = bandwidth.square().reciprocal()
        bias = -0.5 * (k.square() * metric).sum(-1, keepdim=True) * inverse_sq
        q = q * metric * inverse_sq
    hi = bias.to(fmt)
    bias_lanes = (hi, (bias - hi.to(bias.dtype)).to(fmt)) if split_bias else (hi,)
    q = torch.cat((q.to(fmt), *(torch.ones_like(q[..., :1], dtype=fmt) for _ in bias_lanes)), -1)
    k = torch.cat((k.to(fmt), *bias_lanes), -1)
    aligned = (head + 8) // 8 * 8
    assert aligned >= q.shape[-1]
    q, k = (F.pad(x, (0, aligned - x.shape[-1])) for x in (q, k))
    v = F.pad(v, (0, aligned - head))
    z = F.scaled_dot_product_attention(q, k, v, is_causal=not last_only, scale=1.0)[..., :head]
    return attention.output(z.transpose(1, 2).contiguous().view(batch, q.shape[-2], attention.d_model)), None


def variants(checkpoint):
    models = make_variants(checkpoint, "cuda")
    del models["augmented_math_fp32"]
    experiments = (("split", False, True), ("balanced", True, False), ("balanced_split", True, True))
    for name, balanced, split_bias in experiments:
        model = deepcopy(models["augmented_flash_bf16"])
        for block in model.blocks:
            forward = partial(stable_forward, balanced=balanced, split_bias=split_bias)
            block.context.forward = MethodType(forward, block.context)
        models[name + "_flash_bf16"] = model
    return models


def run(args):
    torch.set_float32_matmul_precision("highest")
    checkpoint_hash = sha256(args.checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash == args.checkpoint_sha256
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["arm"] == "radial_without_position" and checkpoint["hide_query_clock"]
    models = variants(checkpoint)
    timing = {}
    for i, (depth, batch) in enumerate(((128, 16), (512, 16), (512, 32))):
        data, stamp = fixture(22391 + depth, batch, depth)
        order = tuple(models)
        row = {}
        for mode in (*order[i:], *order[:i]):
            reset(models[mode], checkpoint)
            row[mode] = benchmark(models[mode], mode, data, 24)
        timing[f"{depth}/{batch}"] = {"input_sha256": stamp, "steps": row}
    seeds = {}
    for seed in SEEDS:
        gradients = {}
        for depth in DEPTHS:
            data, stamp = fixture(seed + depth, 16, depth)
            for net in models.values():
                reset(net, checkpoint)
            gradients[str(depth)] = {"input_sha256": stamp,
                                     "measurements": gradient_probe(models, data, reference="eager_fp32")}
        batches = [fixture(seed + 102401 + i, 16, DEPTHS[i % 2]) for i in range(128)]
        continuation = {}
        for mode, net in models.items():
            reset(net, checkpoint)
            continuation[mode] = continue_training(net, mode, [data for data, _ in batches])
        held = {}
        for depth in DEPTHS:
            data, stamp = fixture(seed + 827139 + depth, 512, depth)
            held[str(depth)] = {"input_sha256": stamp, "outcomes": assess(models, data)}
        batch_hash = sha256("".join(stamp for _, stamp in batches).encode()).hexdigest()
        seeds[str(seed)] = {"gradients": gradients, "updates_sha256": batch_hash, "continuation": continuation,
                            "held_out": held}
    return {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "checkpoint_sha256": checkpoint_hash, "timing": timing, "seeds": seeds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
