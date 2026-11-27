#!/usr/bin/env python3
"""Use exact-precision fused attention for the hard-clock/rank serving baseline."""

import argparse
from copy import deepcopy
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from types import MethodType

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_radial_precision import stable_forward
from benchmark_kam_radial_sdpa import augmented_forward, labels
from benchmark_kam_typed_guard_cropping import PANELS, refit, terminal
from run_kam_bitemporal_decoder import encode, fingerprint, make_decoder, record_order
from run_kam_bitemporal_noise import generate
from run_kam_packed_clock_inference import EAGER, METHODS, evaluate, modify_unrelated, profile, read
from run_kam_typed_clock_guard import guarded_tokens


def dot_sdpa(attention, inputs, return_weights=False):
    assert not return_weights and attention.dropout.p == 0
    assert attention.relative_bias is None and attention.window is None and attention.score.score_type == "dot"
    n, length, _ = inputs.shape
    qkv = attention.qkv(inputs).view(n, length, 3, attention.num_heads, attention.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attention.score._prepare(q, k)
    z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return attention.output(z.transpose(1, 2).contiguous().view(n, length, attention.d_model)), None


def backend_call(model, kernel, data, positions, backend, bf16=False):
    assert kernel == EAGER
    with sdpa_kernel(backend), torch.autocast("cuda", torch.bfloat16, enabled=bf16):
        hidden, _ = model.encode(data)
        return model.readout(hidden[torch.arange(len(data), device=data.device), positions])


def variants(geometry, original):
    fused = deepcopy(original)
    for block in fused.blocks:
        f = dot_sdpa if geometry == "dot" else augmented_forward
        block.context.forward = MethodType(f, block.context)
    flash = deepcopy(fused)
    if geometry == "radial":
        for block in flash.blocks:
            f = partial(stable_forward, balanced=True, split_bias=True)
            block.context.forward = MethodType(f, block.context)
    return {
        "eager": (original, read),
        "eager_terminal": (original, terminal),
        "fp32_efficient": (fused, partial(backend_call, backend=SDPBackend.EFFICIENT_ATTENTION)),
        "bf16_flash": (flash, partial(backend_call, backend=SDPBackend.FLASH_ATTENTION, bf16=True)),
    }


@torch.inference_mode()
def measure(a, state):
    stamp = state["reference_sha256"]
    rng = Random(a.seed + 383055)
    preprocess = partial(guarded_tokens, ranked=True)
    panels = {}
    for geometry, weights in state["models"].items():
        model = make_decoder(f"{geometry}_without_position", a.seed, "cuda").eval()
        model.load_state_dict(weights)
        options = variants(geometry, model)
        unsupported = {}
        curves = {}
        for depth, width in PANELS:
            seed = a.seed + depth + 127830
            scene = generate(seed, 64, depth, "mixed", "cpu")
            varied = modify_unrelated(scene, a.seed + 110639 + depth)
            perm = record_order(64, depth, a.seed + depth + 121575, "cpu")
            for name, chosen in (("standard", scene), ("future_distractors", varied)):
                raw, entity, value = encode(chosen, perm, hide_query_clock=True)
                raw = raw.cuda()
                truth = torch.stack((entity, value), -1).cuda()
                base = torch.cat([evaluate(model, EAGER, x, "full", preprocess=preprocess) for x in raw.split(width)])
                rows = {}
                configs = [(impl, method) for impl in options for method in METHODS]
                for impl, method in rng.sample(configs, len(configs)):
                    net, reader = options[impl]
                    if impl in unsupported:
                        continue
                    call = partial(evaluate, net, EAGER, method=method, preprocess=preprocess, readout=reader)
                    try:
                        logits = torch.cat([call(x).float() for x in raw.split(width)])
                    except RuntimeError as err:
                        if impl not in {"fp32_efficient", "bf16_flash"} or "No viable backend" not in str(err):
                            raise
                        unsupported[impl] = str(err)
                        continue
                    assert torch.isfinite(logits).all()
                    choices = labels(logits)
                    start = (a.seed % (len(raw)//width)) * width
                    rows[f"{impl}/{method}"] = {"decisions": choices.tolist(),
                                                "matches_eager": int(choices.eq(labels(base)).all(-1).sum()),
                                                "strict": int(choices.eq(truth).all(-1).sum()),
                                                "max_logit_error": float((logits-base).abs().max()),
                                                "timing": profile(lambda: call(raw[start:start+width]), a.repeats)}
                curves[f"{depth}/{name}"] = {"batch": width, "fixture_sha256": fingerprint(chosen, perm), "arms": rows}
        panels[geometry] = {"panels": curves, "unsupported": unsupported}
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "weights_sha256": sha256(a.weights.read_bytes()).hexdigest(), "reference_sha256": stamp, "geometries": panels}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--refit-reference", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--repeats", type=int, default=20)
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    if a.refit_reference:
        assert a.output is None and not a.weights.exists()
        models = {name: {k: v.cpu() for k, v in model.state_dict().items()}
                  for name, model in refit(a.seed, a.refit_reference)}
        torch.save({"seed": a.seed, "models": models,
                    "reference_sha256": sha256(a.refit_reference.read_bytes()).hexdigest()}, a.weights)
    else:
        if a.output is None:
            p.error("Supply --output for an inference run")
        state = torch.load(a.weights, map_location="cpu", weights_only=True)
        assert state["seed"] == a.seed
        a.output.write_text(json.dumps(measure(a, state), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
