#!/usr/bin/env python3
"""Benchmark a FlashAttention adapter for learned relative-bias-free radial KAM."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from types import MethodType

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from run_kam_bitemporal_decoder import encode, fingerprint, make_decoder, pack, record_order
from run_kam_bitemporal_noise import generate


def augmented_forward(attention, inputs, return_weights=False):
    assert not attention.training and not return_weights
    assert attention.relative_bias is None and attention.window is None
    score = attention.score
    assert score.score_type == "rbf"
    batch, length, _ = inputs.shape
    projected = attention.qkv(inputs).view(batch, length, 3, attention.num_heads, attention.head_dim)
    q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = score._prepare(q, k)
    q, k = score._prepare(q, k)  # Matches the existing expanded radial call path.
    metric = score.metric()[None, :, None, :]
    inverse_bandwidth_sq = score.bandwidth()[None, :, None, None].square().reciprocal()
    key_bias = -0.5 * (k.square() * metric).sum(-1, keepdim=True) * inverse_bandwidth_sq
    q = torch.cat(((q * metric * inverse_bandwidth_sq).to(k.dtype), torch.ones_like(k[..., :1])), -1)
    k = torch.cat((k, key_bias.to(k.dtype)), -1)
    alignment = (-q.shape[-1]) % 8
    q = F.pad(q, (0, alignment))
    k = F.pad(k, (0, alignment))
    v = F.pad(v, (0, alignment + 1))
    result = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)[..., :attention.head_dim]
    result = result.transpose(1, 2).contiguous().view(batch, length, attention.d_model)
    return attention.output(result), None


def make_variants(checkpoint, device):
    models = {}
    for name in ("eager_fp32", "eager_bf16", "augmented_math_fp32", "augmented_flash_bf16"):
        model = make_decoder(checkpoint["arm"], checkpoint["seed"], device).eval()
        model.load_state_dict(checkpoint["weights"])
        if name.startswith("augmented"):
            for block in model.blocks:
                block.context.forward = MethodType(augmented_forward, block.context)
        models[name] = model
    return models


def predict(model, name, inputs):
    with torch.autocast("cuda", torch.bfloat16, enabled=name.endswith("bf16")):
        if name == "augmented_flash_bf16":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return model(inputs)
        if name == "augmented_math_fp32":
            with sdpa_kernel(SDPBackend.MATH):
                return model(inputs)
        return model(inputs)


def labels(logits):
    return torch.stack((logits[:, :64].argmax(-1), logits[:, 64:].argmax(-1)), -1)


def timing(call, batch, repeats):
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        call()
        end.record()
    torch.cuda.synchronize()
    history = sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    p50 = median(history)
    return {"p50_ms": p50, "p95_ms": history[int(.95 * (len(history) - 1))],
            "episodes_per_second_at_p50": batch / (p50 / 1000),
            "peak_allocated_increment": torch.cuda.max_memory_allocated() - before}


def graph_timing(model, name, x, alternate, repeats):
    static_inputs = x.clone()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            predict(model, name, static_inputs)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    before = torch.cuda.memory_reserved()
    with torch.cuda.graph(graph):
        captured = predict(model, name, static_inputs)
    graph.replay()
    original = predict(model, name, static_inputs)
    torch.testing.assert_close(captured, original, rtol=0, atol=0)
    static_inputs.copy_(alternate)
    graph.replay()
    changed = predict(model, name, static_inputs)
    torch.testing.assert_close(captured, changed, rtol=0, atol=0)
    reserved = torch.cuda.memory_reserved() - before
    return {"net_cuda_reserved_change_after_capture": reserved,
            **timing(graph.replay, len(x), repeats)}


@torch.inference_mode()
def execute(cfg):
    assert cfg.repeats >= 20
    torch.set_float32_matmul_precision("highest")
    state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    assert state["arm"] == "radial_without_position" and state["hide_query_clock"]
    models = make_variants(state, "cuda")
    result = {}
    for depth in (20, 128, 512):
        scene = generate(cfg.seed, cfg.examples, depth, "mixed", "cuda")
        order = record_order(cfg.examples, depth, cfg.seed + 11109, "cuda")
        inputs, entities, payloads = encode(scene, order, hide_query_clock=True)
        truth = torch.stack((entities, payloads), -1)
        reference = torch.cat([predict(models["eager_fp32"], "eager_fp32", x) for x in inputs.split(32)])
        ref_labels = labels(reference)
        depth_report = {"examples": len(inputs), "fixture_sha256": fingerprint(scene, order), "modes": {}}
        for mode, net in models.items():
            logits = torch.cat([predict(net, mode, x) for x in inputs.split(32)]).float()
            prediction = labels(logits)
            error = (logits - reference).abs()
            if mode == "augmented_math_fp32":
                torch.testing.assert_close(logits, reference, atol=0.00015, rtol=0.00015)
            mode_report = {"joint": int(prediction.eq(truth).all(-1).sum()),
                           "joint_agreement": int(prediction.eq(ref_labels).all(-1).sum()),
                           "max_logit_error": float(error.max()), "mean_logit_error": float(error.mean()),
                           "timing": {str(n): timing(lambda: predict(net, mode, inputs[:n]), n, cfg.repeats)
                                      for n in (1, 16, 64)}}
            if cfg.cuda_graph and mode in ("eager_fp32", "augmented_flash_bf16"):
                mode_report["graph"] = {str(n): graph_timing(net, mode, inputs[:n], inputs[-n:], cfg.repeats)
                                        for n in (1, 16, 64)}
            depth_report["modes"][mode] = mode_report
        result[str(depth)] = depth_report
    binding = sha256(cfg.checkpoint.read_bytes()).hexdigest()
    report = {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "seed": cfg.seed,
              "checkpoint_sha256": binding, "checkpoint_train_tokens": state["train_tokens"],
              "event_repeats": cfg.repeats, "depths": result}
    cfg.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=8519227)
    p.add_argument("--examples", type=int, default=512)
    p.add_argument("--repeats", type=int, default=40)
    p.add_argument("--cuda-graph", action="store_true")
    cfg = p.parse_args()
    if cfg.examples < (128 if cfg.cuda_graph else 64):
        p.error("Use at least 64 examples for timing or 128 for disjoint CUDA graph validation")
    execute(cfg)


if __name__ == "__main__":
    main()
