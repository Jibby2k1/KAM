#!/usr/bin/env python3
"""Train the matched 2,048-byte KAM and dot models on one frozen corpus."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from kam.memory import sparse_kam
from kam.transformer.baselines import build_baseline

PAIR = "wiki_code_arithmetic_2048:731109"
ARMS = ("T-KAM-F", "T-WIDE")
DOMAINS = ("wiki", "code", "math")
CONTEXT = 2048
BATCHES = (4, 2, 2)
DEFAULT_STEPS = 8192


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def row_for(arm, corpus_sha, steps):
    return {
        "architecture": arm, "row_id": f"long-v1-{arm}", "pair_id": PAIR,
        "corpus_id": "wiki_code_arithmetic_2048", "corpus_sha256": corpus_sha,
        "training_seed": 731109, "data_seed": 731110,
        "minimum_tokens": steps * sum(BATCHES) * CONTEXT,
        "steps": steps, "batch_size": sum(BATCHES), "sequence_length": CONTEXT,
        "scale": "10M", "target_parameter_budget": 10_000_000,
        "num_supports": 1024, "top_k": 4, "precision": "bf16",
    }


def model_for(row):
    model, _ = build_baseline(
        row["architecture"], scale=row["scale"], vocab_size=256, max_seq_len=CONTEXT,
        num_supports=row["num_supports"], top_k=row["top_k"],
        seed=row["training_seed"], target_parameters=row["target_parameter_budget"],
    )
    return model.cuda()


def omit_statistics(*_args, **_kwargs):
    return {}


def corpus_tensors(root, manifest):
    data = {}
    for split in ("train", "validation", "test"):
        data[split] = []
        for domain in DOMAINS:
            name = f"{domain}-{split}"
            path = root / "corpus" / f"{name}.bin"
            assert digest(path) == manifest["corpus"][name]["sha256"]
            data[split].append(torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).cuda())
    return data


def sample_plan(data, steps, seed):
    rng = torch.Generator().manual_seed(seed)
    return [torch.randint(0, len(tokens) - CONTEXT, (steps, count), generator=rng)
            for tokens, count in zip(data, BATCHES, strict=True)]


def batch(data, starts, offsets):
    packed = [tokens[index[:, None] + offsets].long()
              for tokens, index in zip(data, starts, strict=True)]
    return torch.cat(packed, dim=0)


def losses(model, packed):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(packed[:, :-1])
    nll = F.cross_entropy(logits.float().flatten(0, 1), packed[:, 1:].flatten(), reduction="none")
    return nll.view(sum(BATCHES), CONTEXT)


@torch.inference_mode()
def evaluate(model, data, offsets, seed):
    model.eval()
    plan = [index.cuda() for index in sample_plan(data, 16, seed)]
    totals = torch.zeros(3, device="cuda")
    for i in range(16):
        nll = losses(model, batch(data, [index[i] for index in plan], offsets))
        totals += torch.stack([part.mean() for part in nll.split(BATCHES)])
    values = dict(zip(DOMAINS, (totals / 16).tolist(), strict=True))
    values["mixed"] = sum(values[d] * n for d, n in zip(DOMAINS, BATCHES, strict=True)) / sum(BATCHES)
    model.train()
    return values


def save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(root, arm, steps):
    corpus_manifest = root / "corpus" / "manifest.json"
    rows = [row_for(name, digest(corpus_manifest), steps) for name in ARMS]
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    assert (root / "manifest.jsonl").read_text() == manifest_text
    row = next(row for row in rows if row["architecture"] == arm)
    torch.manual_seed(row["training_seed"])
    torch.backends.mha.set_fastpath_enabled(False)
    sparse_kam.routing_diagnostics = omit_statistics
    model = model_for(row)
    parameters = sum(p.numel() for p in model.parameters())
    assert abs(parameters / row["target_parameter_budget"] - 1) < .01
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=.1, fused=True)
    data = corpus_tensors(root, json.loads(corpus_manifest.read_text()))
    plan = sample_plan(data["train"], steps, row["data_seed"])
    exposure = hashlib.sha256(b"".join(index.numpy().tobytes() for index in plan)).hexdigest()
    plan = [idx.cuda() for idx in plan]
    offsets = torch.arange(CONTEXT + 1, device="cuda")
    base = root / "checkpoints" / f"{row['row_id']}_seed{row['training_seed']}"
    restart = base.with_name(base.name + "_restart.pt")
    begin = 0
    history = []
    elapsed_before = 0.0
    if restart.is_file():
        state = torch.load(restart, map_location="cuda", weights_only=True)
        assert state["row"] == row and state["exposure_sha256"] == exposure
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        begin, history, elapsed_before = state["step"], state["history"], state["elapsed"]
    print(arm, parameters, row["minimum_tokens"], torch.cuda.get_device_name(), exposure, flush=True)
    started = time.perf_counter()
    model.train()
    for idx in range(begin, steps):
        step = idx + 1
        progress = max(0.0, (step - 256) / max(steps - 256, 1))
        lr = 4e-4 * min(step / 256, 1.0) * (.1 + .9 * (1 + math.cos(math.pi * progress)) / 2)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        nll = losses(model, batch(data["train"], [s[idx] for s in plan], offsets)).mean()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 4 or step % 256 == 0 or step == steps:
            print(arm, step, float(nll.detach()), round(elapsed_before + time.perf_counter() - started, 3), flush=True)
        if step % 1024 == 0 or step == steps:
            valid = evaluate(model, data["validation"], offsets, 91901)
            elapsed = elapsed_before + time.perf_counter() - started
            history.append({"step": step, "validation": valid, "elapsed_s": elapsed})
            print("validation", valid, flush=True)
            save({"row": row, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "step": step, "history": history, "elapsed": elapsed, "exposure_sha256": exposure}, restart)
    valid = history[-1]["validation"]
    test = evaluate(model, data["test"], offsets, 91902)
    metrics = {
        "tokens": row["minimum_tokens"], "validation_loss": valid["mixed"],
        "validation_nll_by_domain": valid, "test_nll_by_domain": test, "parameters": parameters,
        "exposure_sha256": exposure, "history": history, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
    }
    final = base.with_name(base.name + "_final.pt")
    save({"row": row, "model": model.state_dict(), "metrics": metrics}, final)
    metrics["checkpoint_sha256"] = digest(final)
    (root / f"train-{arm}.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    if args.arm:
        train(args.root, args.arm, args.steps)
    else:
        rows = [row_for(arm, digest(args.root / "corpus" / "manifest.json"), args.steps) for arm in ARMS]
        (args.root / "manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
