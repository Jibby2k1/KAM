#!/usr/bin/env python3
"""Compare peak learning rates and a forward KL to the frozen backbone."""
import argparse
import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from kam_adapter_preservation import (
    ARMS, SEEDS, TRAINING_RECIPES, digest, key, layers_for, load_registered_model, model_identity,
)
from kam_pretrained_adapters import AdapterStack, mixed_precision
from kam_shared_teacher_prefix import reuse_prefix
from train_kam_pretrained import BATCHES, DOMAINS, LENGTH, batch, corpus, layout, save

STEPS = 256
KL_STRIDE = 4


def teacher_cache_identity(root, model, seed, exposure):
    model_id, revision, _ = model_identity(root)
    return {
        "model": model_id, "revision": revision, "seed": seed, "exposure_sha256": exposure,
        "model_fingerprints": (root / "model-sha256.txt").read_text(),
        "token_manifest_sha256": digest(root / "token-manifest.json"),
        "shape": (STEPS, sum(BATCHES), LENGTH // KL_STRIDE, model.config.hidden_size),
        "sequence_length": LENGTH, "kl_stride": KL_STRIDE, "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
    }


def optimizer_fingerprint(parameters, optimizer):
    state = hashlib.sha256()
    for parameter in parameters:
        values = optimizer.state[parameter]
        for tensor in (parameter, values["step"], values["exp_avg"], values["exp_avg_sq"]):
            state.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return state.hexdigest()


def losses(model, stack, packed, penalty, phase, *, teacher_hidden=None):
    if penalty and teacher_hidden is None:
        stack.active = False
        try:
            with torch.no_grad(), mixed_precision():
                ref = model.model(input_ids=packed[:, :-1], use_cache=False).last_hidden_state
                ref_logits = model.lm_head(ref[:, phase::KL_STRIDE]).float()
                ref_log_p = ref_logits.log_softmax(-1)
        finally:
            stack.active = True
    elif penalty:
        with torch.no_grad(), mixed_precision():
            ref_log_p = model.lm_head(teacher_hidden).float().log_softmax(-1)
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


def train(root, seed, recipe, arm, teacher_cache=None, shared_teacher_prefix=False):
    peak, penalty = TRAINING_RECIPES[recipe]
    if teacher_cache is not None or shared_teacher_prefix:
        if not penalty:
            raise ValueError("Teacher reuse requires a positive KL weight")
        torch.use_deterministic_algorithms(True)
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
    if shared_teacher_prefix:
        identity["shared_teacher_prefix_blocks"] = layers[0] + 1
    tape, cache_setup = None, None
    if teacher_cache is not None:
        setup_started = time.perf_counter()
        cached = torch.load(teacher_cache, map_location="cuda", weights_only=True)
        expected = teacher_cache_identity(root, model, seed, exposure)
        assert cached["identity"] == expected, "teacher cache does not match the model, GPU or fit schedule"
        tape = cached["activations"]
        assert tape.dtype == torch.bfloat16 and tuple(tape.shape) == expected["shape"]
        workspace = tape.new_empty((sum(BATCHES), LENGTH, model.config.hidden_size))
        stack.active = False
        try:
            with torch.inference_mode(), mixed_precision():
                for phase in range(KL_STRIDE):
                    packed = batch(data, [p[phase] for p in plan], offsets)
                    current = model.model(input_ids=packed[:, :-1], use_cache=False).last_hidden_state
                    torch.testing.assert_close(tape[phase], current[:, phase::KL_STRIDE], rtol=0, atol=0)
        finally:
            stack.active = True
        cache_setup = time.perf_counter() - setup_started
        identity["teacher_cache_identity"] = expected
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
        phase = i % KL_STRIDE
        teacher_hidden = None
        if tape is not None:
            teacher_hidden = workspace[:, phase::KL_STRIDE]
            teacher_hidden.copy_(tape[i])
        context = reuse_prefix(model, stack, layers[0]) if shared_teacher_prefix else nullcontext()
        with context:
            ce, kl = losses(model, stack, batch(data, [p[i] for p in plan], offsets), penalty, phase,
                            teacher_hidden=teacher_hidden)
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
    if tape is not None:
        details["cache_load_and_check_s"] = cache_setup
        details["cached_optimizer_sha256"] = optimizer_fingerprint(stack.modules.parameters(), optim)
    if shared_teacher_prefix:
        details["shared_prefix_optimizer_sha256"] = optimizer_fingerprint(stack.modules.parameters(), optim)
    (root / f"train-{ident}.json").write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--recipe", choices=TRAINING_RECIPES, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--teacher-cache", type=Path)
    reuse.add_argument("--shared-teacher-prefix", action="store_true")
    a = parser.parse_args()
    train(a.root, a.seed, a.recipe, a.arm, a.teacher_cache, a.shared_teacher_prefix)
