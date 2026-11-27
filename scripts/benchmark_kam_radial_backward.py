#!/usr/bin/env python3
"""Benchmark gradients and AdamW steps for the augmented radial attention kernel."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch.nn import functional as F

from benchmark_kam_radial_sdpa import labels, make_variants, predict
from run_kam_bitemporal_decoder import encode, fingerprint, record_order
from run_kam_bitemporal_noise import generate

DEPTHS = (20, 128, 512)
BATCHES = (1, 16, 32)
LR = 0.0005


def fixture(seed, examples, depth):
    scene = generate(seed, examples, depth, "mixed", "cuda")
    order = record_order(examples, depth, seed + 1147, "cuda")
    x, entity, payload = encode(scene, order, hide_query_clock=True)
    return (x, entity, payload), fingerprint(scene, order)


def objective(logits, entities, payloads):
    logits = logits.float()
    return F.cross_entropy(logits[:, :64], entities) + F.cross_entropy(logits[:, 64:], payloads)


def reset(model, checkpoint):
    model.load_state_dict(checkpoint["weights"])
    model.zero_grad(set_to_none=True)
    model.train()


def gradient_probe(models, data, *, reference=None):
    x, entity, payload = data
    gradients, logits, costs = {}, {}, {}
    for mode, net in models.items():
        net.zero_grad(set_to_none=True)
        output = predict(net, mode, x)
        cost = objective(output, entity, payload)
        cost.backward()
        grads = {k: p.grad.detach().float().clone() for k, p in net.named_parameters() if p.grad is not None}
        assert grads and all(torch.isfinite(g).all() for g in grads.values())
        assert torch.isfinite(cost)
        gradients[mode], logits[mode], costs[mode] = grads, output.float().detach(), float(cost.detach())
    results = {}
    for mode, other in gradients.items():
        refmode = reference or ("eager_bf16" if mode == "augmented_flash_bf16" else "eager_fp32")
        base = gradients[refmode]
        ref_logits = logits[refmode]
        error = (logits[mode] - ref_logits).abs()
        assert base.keys() == other.keys()
        families = {
            "all": tuple(base),
            "metric_and_bandwidth": tuple(k for k in base if "raw_metric" in k or "raw_bandwidth" in k),
            "qkv": tuple(k for k in base if ".qkv." in k),
        }
        comparison = {}
        for family, names in families.items():
            assert names
            a = torch.cat([base[k].flatten() for k in names])
            b = torch.cat([other[k].flatten() for k in names])
            comparison[family] = {"relative_l2": float((a - b).norm() / a.norm().clamp_min(1e-12)),
                                  "max_abs": float((a - b).abs().max()),
                                  "cosine": float(F.cosine_similarity(a, b, dim=0))}
        results[mode] = {"reference": refmode, "loss": costs[mode],
                         "loss_delta": costs[mode] - costs[refmode],
                         "max_logit_error": float(error.max()),
                         "fp32_logit_tolerance": torch.allclose(logits[mode], ref_logits, atol=2e-4, rtol=2e-4),
                         "gradient_error": comparison}
    return results


def step(net, mode, opt, data):
    x, entity, payload = data
    opt.zero_grad(set_to_none=True)
    loss = objective(predict(net, mode, x), entity, payload)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return loss.detach()


def benchmark(net, mode, data, repeats):
    opt = torch.optim.AdamW(net.parameters(), lr=LR, fused=True)
    for _ in range(5):
        step(net, mode, opt, data)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for begin, end in zip(starts, ends, strict=True):
        begin.record()
        loss = step(net, mode, opt, data)
        end.record()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    values = sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    p50 = median(values)
    return {"p50_ms": p50, "p95_ms": values[int(.95 * (len(values) - 1))],
            "tokens_per_second": data[0].shape[0] * data[0].shape[1] / (p50 / 1000),
            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated() - resident,
            "final_repeated_loss": float(loss)}


def continue_training(net, mode, updates):
    opt = torch.optim.AdamW(net.parameters(), lr=LR, fused=True)
    torch.cuda.synchronize()
    started = perf_counter()
    mid = []
    tokens = 0
    for index, batch in enumerate(updates, 1):
        loss = step(net, mode, opt, batch)
        tokens += batch[0].shape[0] * batch[0].shape[1]
        if index % 32 == 0:
            assert torch.isfinite(loss)
            mid.append({"step": index, "loss": float(loss)})
    torch.cuda.synchronize()
    return {"steps": len(updates), "tokens": tokens, "elapsed_seconds": perf_counter() - started, "loss": mid}


@torch.inference_mode()
def assess(models, data):
    x, entity, payload = data
    truth = torch.stack((entity, payload), -1)
    preds = {}
    for mode, net in models.items():
        net.eval()
        pred = torch.cat([predict(net, mode, block) for block in x.split(16)])
        preds[mode] = labels(pred)
    ref = preds["eager_fp32"]
    return {mode: {"correct": pred.eq(truth).all(-1).int().tolist(),
                   "agrees_with_fp32": pred.eq(ref).all(-1).int().tolist()} for mode, pred in preds.items()}


def run(a):
    torch.set_float32_matmul_precision("highest")
    checkpoint_hash = sha256(a.checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash == a.checkpoint_sha256
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["arm"] == "radial_without_position" and checkpoint["hide_query_clock"]
    models = make_variants(checkpoint, "cuda")
    probes, performance = {}, {}
    for d, depth in enumerate(DEPTHS):
        test, stamp = fixture(a.seed + depth, 32, depth)
        for net in models.values():
            reset(net, checkpoint)
        small = tuple(d[:16] for d in test)
        probes[str(depth)] = {"input_sha256": stamp, **gradient_probe(models, small)}
        result = {}
        names = tuple(models)
        for b, batch in enumerate(BATCHES):
            shift = (d + b) % len(names)
            for mode in (*names[shift:], *names[:shift]):
                net = models[mode]
                reset(net, checkpoint)
                result.setdefault(str(batch), {})[mode] = benchmark(
                    net, mode, tuple(values[:batch] for values in test), a.repeats)
        performance[str(depth)] = result

    blocks = [fixture(a.seed + 102401 + i, 16, DEPTHS[i % 2]) for i in range(128)]
    updates = [data for data, _ in blocks]
    continued = {}
    for mode, net in models.items():
        reset(net, checkpoint)
        continued[mode] = continue_training(net, mode, updates)
    held = {}
    for depth in DEPTHS:
        data, stamp = fixture(a.seed + 827139 + depth, 512, depth)
        held[str(depth)] = {"examples": 512, "fixture_sha256": stamp, "outcomes": assess(models, data)}
    return {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "seed": a.seed,
            "checkpoint_sha256": checkpoint_hash, "checkpoint_train_tokens": checkpoint["train_tokens"],
            "timing_repeats": a.repeats, "gradients": probes, "training_step": performance,
            "continuation_input_sha256": sha256("".join(stamp for _, stamp in blocks).encode()).hexdigest(),
            "continuation": continued, "held_out": held}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=729127)
    p.add_argument("--repeats", type=int, default=24)
    a = p.parse_args()
    assert a.repeats >= 20
    a.output.write_text(json.dumps(run(a), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
