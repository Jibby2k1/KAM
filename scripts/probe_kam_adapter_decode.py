#!/usr/bin/env python3
"""Profile the existing backbone and adapter caches at fixed decoding shapes."""
import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch

from eval_kam_pretrained import next_logits
from kam_adapter_preservation import ARMS, digest, key, layers_for, load_registered_model, model_identity
from kam_pretrained_adapters import WIDTH, AdapterStack
from train_kam_adapter_preservation import corpus

SHAPES = tuple((length, batch) for batch in (1, 8) for length in (256, 2048, 4096))
DECODE_TOKENS = 32
ARM_ORDER = ("pretrained", *ARMS)


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


@torch.inference_mode()
def request(model, stack, prompt, continuation, masks, trace=None):
    stack.reset()
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    logits, cache = next_logits(model, stack, prompt, masks[0], use_cache=True)
    if trace is not None:
        trace.append(logits)
    torch.cuda.synchronize()
    after_prefill = time.perf_counter()
    for step in range(continuation.size(1)):
        logits, cache = next_logits(model, stack, continuation[:, step:step+1], masks[step+1],
                               cache, use_cache=True)
        if trace is not None:
            trace.append(logits)
    torch.cuda.synchronize()
    done = time.perf_counter()
    backbone = tensor_bytes(t for layer in cache.layers for t in (layer.keys, layer.values))
    adapter = tensor_bytes(t for module in stack.modules if module.cache is not None for t in module.cache)
    extra_peak = torch.cuda.max_memory_allocated() - initial
    stack.reset()
    return {"prefill_ms": 1000 * (after_prefill - start),
            "decode_ms": 1000 * (done - after_prefill),
            "request_ms": 1000 * (done - start),
            "peak_extra_bytes": extra_peak, "backbone_cache_bytes": backbone,
            "adapter_cache_bytes": adapter}


def append_static_cache(module, slots, capacity, keys, values):
    assert module.cache_enabled and not torch.is_grad_enabled()
    old = module.cache[0].size(-2) if module.cache is not None else 0
    new = old + keys.size(-2)
    assert new <= capacity
    if slots[0] is None:
        for i, values_now in enumerate((keys, values)):
            shape = (*values_now.shape[:-2], capacity, values_now.size(-1))
            slots[i] = values_now.new_empty(shape)
    for slot, values_now in zip(slots, (keys, values), strict=True):
        slot[..., old:new, :].copy_(values_now)
    module.cache = tuple(slot[..., :new, :] for slot in slots)
    return old, *module.cache


def profile(root, index, include_static):
    length, batch_size = SHAPES[index]
    model, _ = load_registered_model(root)
    model_id, revision, _ = model_identity(root)
    ident = 731221
    layers = layers_for(model)
    arms = (*ARM_ORDER, "dot_static") if include_static else ARM_ORDER
    stacks = {arm: AdapterStack(model, arm.removesuffix("_static"), seed=ident, layers=layers)
              for arm in arms}
    weights = {}
    for arm in ARMS:
        path = root / f"adapter-{key(ident, 'high_lr_kl', arm)}.pt"
        data = torch.load(path, weights_only=True, map_location="cpu")
        assert (data["identity"]["model"], data["identity"]["revision"], data["identity"]["arm"]) == (
            model_id, revision, arm,
        )
        stacks[arm].modules.load_state_dict(data["adapter"])
        stacks[arm].modules.eval()
        weights[arm] = digest(path)
        if include_static and arm == "dot":
            stacks["dot_static"].modules.load_state_dict(data["adapter"])
            stacks["dot_static"].modules.eval()
    file = root / "token-manifest.json"
    tokens = corpus(root, "validation", json.loads(file.read_text()))[0]
    total = length + DECODE_TOKENS
    if include_static:
        for module in stacks["dot_static"].modules:
            module.append_dot_cache = partial(append_static_cache, module, [None, None], total)
    rng = torch.Generator().manual_seed(84671)
    offsets = torch.randint(0, len(tokens) - max(x[0] for x in SHAPES) - DECODE_TOKENS,
                            (batch_size,), generator=rng).cuda()
    complete = tokens[offsets[:, None] + torch.arange(total, device="cuda")].long()
    prompt = complete[:, :length].contiguous()
    continuation = complete[:, length:].contiguous()
    mask = torch.ones((batch_size, total), device="cuda", dtype=torch.bool)
    masks = [mask[:, :length+i] for i in range(DECODE_TOKENS+1)]
    def activate(arm):
        for other, stack in stacks.items():
            if other != "pretrained":
                stack.active = other == arm

    scores = {arm: [] for arm in arms}
    for trial in range(11):
        rotation = trial % len(arms)
        for arm in arms[rotation:] + arms[:rotation]:
            activate(arm)
            row = request(model, stacks[arm], prompt, continuation, masks)
            nbytes = batch_size * total * len(layers) * 2 * WIDTH * 2
            assert row["adapter_cache_bytes"] == (nbytes if arm.startswith("dot") else 0)
            if trial >= 2:
                scores[arm].append(row)
    assert len({row["backbone_cache_bytes"] for rows in scores.values() for row in rows}) == 1
    parity = None
    if include_static:
        baseline, candidate = [], []
        for arm, trace in zip(("dot", "dot_static"), (baseline, candidate), strict=True):
            activate(arm)
            request(model, stacks[arm], prompt, continuation, masks, trace)
        pairs = list(zip(baseline, candidate, strict=True))
        parity = {"positions": len(pairs) * batch_size,
                  "greedy_matches": sum(int((a.argmax(-1) == b.argmax(-1)).sum()) for a, b in pairs),
                  "max_abs_logit_difference": max(float((a-b).abs().max()) for a, b in pairs)}
    payload = {
        "model": model_id, "revision": revision, "seed": ident, "length": length,
        "batch": batch_size, "decode_tokens": DECODE_TOKENS,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "model_fingerprints": (root / "model-sha256.txt").read_text(), "adapter_sha256": weights,
        "token_manifest_sha256": digest(file), "requests": scores, "static_dot_parity": parity,
    }
    mode = "static-" if include_static else ""
    output = root / f"decode-profile-{mode}{length}-{batch_size}.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(output)
    print(model_id, length, batch_size, "measured", len(scores["kam"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int, choices=range(len(SHAPES)))
    parser.add_argument("--static-dot", action="store_true")
    args = parser.parse_args()
    profile(args.root, args.index, args.static_dot)
