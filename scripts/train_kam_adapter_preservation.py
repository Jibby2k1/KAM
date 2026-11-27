#!/usr/bin/env python3
"""Compare peak learning rates and a forward KL to the frozen backbone."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from kam_adapter_preservation import (
    ARMS, SEEDS, TRAINING_RECIPES, digest, key, layers_for, load_registered_model, model_identity,
)
from kam_pretrained_adapters import AdapterStack, mixed_precision
from train_kam_pretrained import BATCHES, DOMAINS, LENGTH, batch, corpus, layout, save

STEPS = 256
KL_STRIDE = 4


def losses(model, stack, packed, penalty, phase):
    if penalty:
        stack.active = False
        try:
            with torch.no_grad(), mixed_precision():
                ref = model.model(input_ids=packed[:, :-1], use_cache=False).last_hidden_state
                ref_logits = model.lm_head(ref[:, phase::KL_STRIDE]).float()
                ref_log_p = ref_logits.log_softmax(-1)
        finally:
            stack.active = True
    with mixed_precision():
        hidden = model.model(input_ids=packed[:, :-1], use_cache=False).last_hidden_state
        student_logits = model.lm_head(hidden)
        ce = F.cross_entropy(student_logits.flatten(0, 1), packed[:, 1:].reshape(-1))
        if penalty:
            student_log_p = student_logits[:, phase::KL_STRIDE].float().log_softmax(-1)
            kl = F.kl_div(student_log_p, ref_log_p, reduction="batchmean", log_target=True)
            kl = kl / student_log_p.shape[1]
        else:
            kl = torch.zeros((), device=packed.device)
    return ce, kl


@torch.no_grad()
def validation(model, stack, data, offsets):
    plan = [p.cuda() for p in layout(data, 4, 83011)]
    values = {domain: torch.zeros((), device="cuda") for domain in DOMAINS}
    kl = torch.zeros((), device="cuda")
    for i in range(4):
        packed = batch(data, [p[i] for p in plan], offsets)
        _, delta = losses(model, stack, packed, 1.0, i)
        kl += delta
        with mixed_precision():
            hidden = model.model(input_ids=packed[:, :-1], use_cache=False).last_hidden_state
            logits = model.lm_head(hidden)
            nll = F.cross_entropy(logits.flatten(0, 1), packed[:, 1:].reshape(-1),
                                  reduction="none").reshape(sum(BATCHES), LENGTH)
        for domain, group in zip(DOMAINS, nll.split(BATCHES), strict=True):
            values[domain] += group.float().mean()
    return {"nll": {domain: float(value / 4) for domain, value in values.items()},
            "forward_kl": float(kl / 4)}


def train(root, seed, recipe, arm):
    peak, penalty = TRAINING_RECIPES[recipe]
    model_id, revision, _ = model_identity(root)
    model, tokenizer = load_registered_model(root)
    layers = layers_for(model)
    stack = AdapterStack(model, arm, seed=seed, layers=layers)
    ids = tokenizer("The residual output begins at zero.", return_tensors="pt").input_ids.cuda()
    with torch.no_grad(), mixed_precision():
        stack.active = False
        reference = model(input_ids=ids, use_cache=False).logits
        stack.active = True
        torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, reference, rtol=0, atol=0)
    path = root / "token-manifest.json"
    manifest = json.loads(path.read_text())
    data = corpus(root, "train", manifest)
    valid = corpus(root, "validation", manifest)
    plan = layout(data, STEPS, seed + 1)
    exposure = hashlib.sha256(b"".join(p.numpy().tobytes() for p in plan)).hexdigest()
    plan = [p.cuda() for p in plan]
    offsets = torch.arange(LENGTH + 1, device="cuda")
    identity = {
        "arm": arm, "recipe": recipe, "seed": seed, "peak_lr": peak, "kl_weight": penalty,
        "model": model_id, "revision": revision, "layers": layers,
        "train_tokens": STEPS * LENGTH * sum(BATCHES),
        "exposure_sha256": exposure, "token_manifest_sha256": digest(path),
        "model_fingerprints": (root / "model-sha256.txt").read_text(),
        "new_trainable_parameters": sum(p.numel() for p in stack.modules.parameters()),
    }
    optim = torch.optim.AdamW(stack.modules.parameters(), lr=peak, weight_decay=.01, fused=True)
    ident = key(seed, recipe, arm)
    restart = root / f"restart-{ident}.pt"
    start, elapsed_before, history = 0, 0.0, []
    if restart.exists():
        old = torch.load(restart, map_location="cuda", weights_only=True)
        assert old["identity"] == identity
        stack.modules.load_state_dict(old["adapter"])
        optim.load_state_dict(old["optimizer"])
        start, elapsed_before, history = old["step"], old["elapsed_s"], old["history"]
    started = time.perf_counter()
    print(ident, identity["new_trainable_parameters"], identity["train_tokens"], torch.cuda.get_device_name(), flush=True)
    for i in range(start, STEPS):
        step = i + 1
        progress = max(0, (step - 16) / (STEPS - 16))
        optim.param_groups[0]["lr"] = peak * min(step / 16, 1) * (.1 + .45 * (1 + math.cos(math.pi * progress)))
        optim.zero_grad(set_to_none=True)
        ce, kl = losses(model, stack, batch(data, [p[i] for p in plan], offsets), penalty, i % KL_STRIDE)
        objective = ce + penalty * kl
        objective.backward()
        torch.nn.utils.clip_grad_norm_(stack.modules.parameters(), 1.0, error_if_nonfinite=True)
        optim.step()
        if step == 4 or step % 64 == 0:
            print(ident, step, float(ce.detach()), float(kl.detach()), flush=True)
        if step % 128 == 0:
            val = validation(model, stack, valid, offsets)
            elapsed = elapsed_before + time.perf_counter() - started
            history.append({"step": step, "elapsed_s": elapsed, **val})
            save(restart, {"identity": identity, "adapter": stack.modules.state_dict(),
                           "optimizer": optim.state_dict(), "step": step, "elapsed_s": elapsed, "history": history})
    adapter = root / f"adapter-{ident}.pt"
    save(adapter, {"identity": identity, "adapter": stack.modules.state_dict()})
    details = {**identity, "training": history, "adapter_sha256": digest(adapter),
               "gpu": torch.cuda.get_device_name(), "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    (root / f"train-{ident}.json").write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--recipe", choices=TRAINING_RECIPES, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    a = parser.parse_args()
    train(a.root, a.seed, a.recipe, a.arm)
