#!/usr/bin/env python3
"""Train only exactly matched adapters on a frozen Qwen backbone."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from kam_pretrained_adapters import AdapterStack, MODEL_ID, REVISION, load_model, mixed_precision

STEPS = 512
LENGTH = 512
BATCHES = (4, 2, 2)
DOMAINS = ("wiki", "code", "math")
SEED = 731202


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def corpus(root, split, manifest):
    data = []
    for domain in DOMAINS:
        name = f"{domain}-{split}"
        path = root / f"{name}.u32"
        assert sha(path) == manifest[name]["sha256"]
        data.append(torch.from_numpy(np.fromfile(path, dtype="<u4").astype("int32")).cuda())
    return data


def layout(data, steps, seed):
    rng = torch.Generator().manual_seed(seed)
    return [torch.randint(0, len(tokens) - LENGTH, (steps, count), generator=rng)
            for tokens, count in zip(data, BATCHES, strict=True)]


def batch(data, positions, offsets):
    return torch.cat([tokens[starts[:, None] + offsets].long()
                      for tokens, starts in zip(data, positions, strict=True)])


def loss(model, packed):
    with mixed_precision():
        hidden = model.model(input_ids=packed[:, :-1], use_cache=False).last_hidden_state
        logits = model.lm_head(hidden)
    return F.cross_entropy(logits.flatten(0, 1), packed[:, 1:].reshape(-1), reduction="none").view(-1, LENGTH)


@torch.no_grad()
def validate(model, data, offsets, seed):
    plan = [idx.cuda() for idx in layout(data, 8, seed)]
    scores = torch.zeros(3, device="cuda")
    for step in range(8):
        nll = loss(model, batch(data, [idx[step] for idx in plan], offsets)).float()
        scores += torch.stack([part.mean() for part in nll.split(BATCHES)])
    return dict(zip(DOMAINS, (scores / 8).tolist(), strict=True))


def save(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def train(root, arm):
    model, tokenizer = load_model(root)
    ids = tokenizer("An adapter must preserve the starting model.", return_tensors="pt").input_ids.cuda()
    with torch.no_grad(), mixed_precision():
        reference = model(input_ids=ids, use_cache=False).logits
        stack = AdapterStack(model, arm)
        initial = model(input_ids=ids, use_cache=False).logits
        torch.testing.assert_close(initial, reference, rtol=0, atol=0)
    token_manifest = root / "token-manifest.json"
    manifest = json.loads(token_manifest.read_text())
    train_data, valid_data, test_data = (corpus(root, name, manifest)
                                         for name in ("train", "validation", "test"))
    train_plan = layout(train_data, STEPS, SEED)
    exposure = hashlib.sha256(b"".join(t.numpy().tobytes() for t in train_plan)).hexdigest()
    train_plan = [t.cuda() for t in train_plan]
    offsets = torch.arange(LENGTH + 1, device="cuda")
    identity = {
        "arm": arm, "model": MODEL_ID, "revision": REVISION,
        "model_fingerprints": (root / "model-sha256.txt").read_text(),
        "token_manifest_sha256": sha(token_manifest), "exposure_sha256": exposure,
        "train_tokens": STEPS * LENGTH * sum(BATCHES),
        "new_trainable_parameters": sum(p.numel() for p in stack.modules.parameters()),
    }
    baseline_test = validate(model, test_data, offsets, 9145)
    optimizer = torch.optim.AdamW(stack.modules.parameters(), lr=8e-4, weight_decay=.01, fused=True)
    restart = root / f"restart-{arm}.pt"
    start, elapsed_before, history = 0, 0, []
    if restart.is_file():
        state = torch.load(restart, map_location="cuda", weights_only=True)
        assert state["identity"] == identity
        stack.modules.load_state_dict(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start, elapsed_before, history = state["step"], state["elapsed_s"], state["history"]
    started = time.perf_counter()
    print(identity, torch.cuda.get_device_name(), flush=True)
    for index in range(start, STEPS):
        step = index + 1
        progress = max(0, (step - 32) / (STEPS - 32))
        optimizer.param_groups[0]["lr"] = 8e-4 * min(step / 32, 1) * (.1 + .45 * (1 + math.cos(math.pi * progress)))
        optimizer.zero_grad(set_to_none=True)
        nll = loss(model, batch(train_data, [p[index] for p in train_plan], offsets)).float().mean()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(stack.modules.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 4 or step % 32 == 0:
            print(arm, step, float(nll.detach()), elapsed_before + time.perf_counter() - started, flush=True)
        if step % 128 == 0:
            validation = validate(model, valid_data, offsets, 9144)
            elapsed = elapsed_before + time.perf_counter() - started
            history.append({"step": step, "validation_nll": validation, "elapsed_s": elapsed})
            save(restart, {"identity": identity, "adapter": stack.modules.state_dict(), "optimizer": optimizer.state_dict(),
                           "step": step, "history": history, "elapsed_s": elapsed})
    target = root / f"adapter-{arm}.pt"
    save(target, {"identity": identity, "adapter": stack.modules.state_dict()})
    results = {**identity, "training": history, "test_nll": validate(model, test_data, offsets, 9145),
               "untouched_test_nll": baseline_test, "adapter_sha256": sha(target),
               "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
               "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    (root / f"train-{arm}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--arm", choices=("kam", "dot"), required=True)
    args = parser.parse_args()
    train(args.root, args.arm)
