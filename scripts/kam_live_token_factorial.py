#!/usr/bin/env python3
"""Matched live-token adapters with a trainable linear or SiLU query branch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from kam_adapter_preservation import LLAMA_ID, SEEDS, digest, layers_for, load_registered_model, model_identity
from kam_math_kl_protocol import write_json
from kam_pretrained_adapters import AdapterStack, HEADS, ResidualAdapter, WIDTH, mixed_precision
from kam_shared_teacher_prefix import reuse_prefix
from train_kam_adapter_preservation import losses
from train_kam_pretrained import BATCHES, LENGTH, batch, corpus, layout, save

PROTOCOL = "kam-live-token-factorial-v1"
ARMS = ("soft-linear", "soft-silu", "top4-linear", "top4-silu")
RUNS = tuple((seed, arm) for seed in SEEDS for arm in ARMS)
STEPS = 256
CHECKPOINTS = (64, 128, 192, 256)
BOTTLENECK = 16
PEAK_LR = 8e-5
KL = .5
GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"


class LiveTokenAdapter(ResidualAdapter):
    def __init__(self, hidden, arm, seed):
        super().__init__(hidden, "dot", seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 100_000)
            self.query_in = nn.Linear(WIDTH, BOTTLENECK, bias=False)
            self.query_out = nn.Linear(BOTTLENECK, WIDTH, bias=False)
            nn.init.zeros_(self.query_out.weight)
        self.register_buffer("silu", torch.tensor(arm.endswith("silu")), persistent=False)
        self.register_buffer("sparse", torch.tensor(arm.startswith("top4")), persistent=False)

    def forward(self, x):
        if self.cache_enabled or self.cache is not None:
            raise ValueError("The matched diagnostic uses uncached causal prefill")
        z = self.down(F.rms_norm(x, (x.shape[-1],)))
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        branch = self.query_in(z)
        q = q + self.query_out(torch.where(self.silu, F.silu(branch), branch))
        q, k, v = (r.unflatten(-1, (HEADS, WIDTH // HEADS)).transpose(1, 2) for r in (q, k, v))
        steps = torch.arange(q.size(-2), device=q.device)
        allowed = steps[:, None] >= steps[None, :]
        if self.mask is not None:
            allowed = allowed & self.mask[:, None, None, :]
        with torch.autocast(x.device.type, enabled=False):
            scores = (q.float() @ k.float().transpose(-1, -2)) * (q.size(-1) ** -.5)
            scores = scores.masked_fill(~allowed, -torch.inf)
            _, indices = scores.topk(min(4, scores.size(-1)), dim=-1)
            selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)
            permitted = allowed & (~self.sparse | selected)
            scores = scores.masked_fill(~permitted, -torch.inf).masked_fill(~allowed.any(-1, keepdim=True), 0.)
            weights = scores.softmax(-1).masked_fill(~permitted, 0.)
            value = (weights @ v.float()).to(q.dtype)
        return x + self.up(self.out(value.transpose(1, 2).reshape_as(z))).to(x.dtype)


class LiveTokenStack(AdapterStack):
    def __init__(self, model, seed, arm):
        super().__init__(model, "pretrained")
        self.active = True
        if arm == "pretrained":
            return
        for index in layers_for(model):
            adapter = LiveTokenAdapter(model.config.hidden_size, arm, seed + index)
            self.modules.append(adapter)
            self.handles.append(model.model.layers[index].register_forward_hook(self.hook(adapter)))
        self.modules.cuda()


def fingerprint(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode() + value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(root, work, index):
    protocol = work / "training.json"
    locked = json.loads(protocol.read_text())
    assert locked["protocol"] == PROTOCOL and locked["steps"] == STEPS and tuple(locked["arms"]) == ARMS
    seed, arm = RUNS[index]
    name = f"{seed}-{arm}"
    model_id, revision, _ = model_identity(root)
    assert model_id == LLAMA_ID and torch.cuda.get_device_name() == GPU
    model, tokenizer = load_registered_model(root)
    stack = LiveTokenStack(model, seed, arm)
    initial_sha256 = fingerprint(stack.modules.state_dict())
    ids = tokenizer("The residual output starts at zero.", return_tensors="pt").input_ids.cuda()
    with torch.no_grad(), mixed_precision():
        stack.active = False
        reference = model(input_ids=ids, use_cache=False).logits
        stack.active = True
        torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, reference, rtol=0, atol=0)
    del reference
    token_manifest = root / "token-manifest.json"
    assert digest(token_manifest) == locked["token_manifest_sha256"]
    data = corpus(root, "train", json.loads(token_manifest.read_text()))
    plan = layout(data, STEPS, seed + 1)
    exposure = hashlib.sha256(b"".join(p.numpy().tobytes() for p in plan)).hexdigest()
    plan = [p.cuda() for p in plan]
    offsets = torch.arange(LENGTH + 1, device="cuda")
    params = list(stack.modules.parameters())
    ident = {"protocol_sha256": digest(protocol), "model": model_id, "revision": revision,
             "model_fingerprints": (root / "model-sha256.txt").read_text(), "seed": seed, "arm": arm,
             "layers": layers_for(model), "exposure_sha256": exposure, "initial_sha256": initial_sha256,
             "parameters": sum(p.numel() for p in params), "tokens_per_step": LENGTH * sum(BATCHES),
             "gpu": GPU, "torch": str(torch.__version__), "cuda": torch.version.cuda}
    optimizer = torch.optim.AdamW(params, lr=PEAK_LR, weight_decay=.01, fused=True)
    restart = work / f"restart-{name}.pt"
    start, prior_time, history = 0, 0., []
    if restart.exists():
        payload = torch.load(restart, map_location="cuda", weights_only=True)
        assert payload["identity"] == ident
        stack.modules.load_state_dict(payload["adapter"])
        optimizer.load_state_dict(payload["optimizer"])
        start, prior_time, history = payload["step"], payload["train_seconds"], payload["history"]
        del payload
    torch.cuda.synchronize()
    began = perf_counter()
    for step in range(start + 1, STEPS + 1):
        progress = max(0, (step - 16) / (STEPS - 16))
        optimizer.param_groups[0]["lr"] = PEAK_LR * min(step / 16, 1) * (.1 + .45 * (1 + math.cos(math.pi * progress)))
        optimizer.zero_grad(set_to_none=True)
        packed = batch(data, [p[step-1] for p in plan], offsets)
        with reuse_prefix(model, stack, ident["layers"][0]):
            ce, kl = losses(model, stack, packed, KL, (step-1) % 4)
        (ce + KL * kl).backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        if step in CHECKPOINTS:
            torch.cuda.synchronize()
            prior_time += perf_counter() - began
            state = stack.modules.state_dict()
            path = work / f"adapter-{name}-{step}.pt"
            save(path, {"identity": ident, "adapter": state, "step": step})
            history.append({"step": step, "train_seconds": prior_time, "train_ce": float(ce.detach()),
                            "train_kl": float(kl.detach()), "adapter_sha256": digest(path)})
            save(restart, {"identity": ident, "adapter": state, "optimizer": optimizer.state_dict(), "step": step,
                           "train_seconds": prior_time, "history": history})
            print(name, history[-1], flush=True)
            began = perf_counter()
    write_json(work / f"train-{name}.json", {"identity": ident, "history": history,
                                            "peak_gpu_bytes": torch.cuda.max_memory_allocated()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("index", type=int, choices=range(len(RUNS)))
    args = p.parse_args()
    train(args.root, args.work, args.index)
