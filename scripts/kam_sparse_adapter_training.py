#!/usr/bin/env python3
"""Continue identical adapters under dense and protected sparse attention."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from audit_kam_natural_data import stable
from eval_kam_backbone_shortlist import load
from eval_kam_long_sparse import language_score, retrieval_score
from kam_adapter_preservation import SEEDS, digest
from kam_live_token_factorial import GPU, PEAK_LR, fingerprint
from kam_math_kl_protocol import write_json
from kam_pretrained_adapters import mixed_precision
from kam_query_window_flash import WindowAttention
from kam_trainable_block_attention import differentiable_sparse
from prepare_kam_sparse_adaptation import PROTOCOL, STEPS
from train_kam_pretrained import save

ARMS = ("flash", "protected")
RUNS = tuple((s,a) for s in SEEDS for a in ARMS)
WIDTHS = (4096, 8192)
TAIL = 256
KL_STRIDE = 16
KL = .5
WARMUP = 8


def read(path):
    return json.loads(path.read_text())


class SparseTrainingAttention(WindowAttention):
    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        if self.mode == "protected" and torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
            if mask is not None or dropout or not module.is_causal or q.size(-2) != k.size(-2):
                raise ValueError("Sparse adaptation uses strictly causal unpadded prefill")
            answer = differentiable_sparse(q, k, v, scaling)
            return answer.transpose(1, 2).contiguous(), None
        return super().forward(module, q, k, v, mask, dropout, scaling, **kwargs)


def samples(work, seed):
    manifest = work / "sparse-train-manifest.json"
    inputs = work / "sparse-train-inputs.json"
    spec = read(manifest)
    assert (spec["protocol"], spec["inputs_sha256"], spec["width"], spec["steps"]) == (
        PROTOCOL, digest(inputs), WIDTHS[0], STEPS)
    candidates = read(inputs)
    allowed = {(r["id"],n) for r in spec["books"] for n in r["starts"]}
    assert len(candidates) == spec["available_windows"] == len(allowed)
    assert all((x["id"],x["start"]) in allowed and len(x["tokens"]) == WIDTHS[0]+1 for x in candidates)
    ordered = sorted(candidates, key=lambda row: stable(PROTOCOL+f":{seed}:{row['id']}:{row['start']}"))[:STEPS]
    windows = torch.tensor([x["tokens"] for x in ordered], dtype=torch.int64)
    identity = hashlib.sha256(windows.numpy().tobytes()).hexdigest()
    return windows, digest(manifest), identity


def identity(work, seed, checkpoint, stack):
    windows, manifest_hash, exposure = samples(work, seed)
    return windows, {"protocol": PROTOCOL, "seed": seed, "gpu": GPU, "source_checkpoint_sha256": checkpoint,
                    "training_manifest_sha256": manifest_hash, "exposure_sha256": exposure,
                    "initial_adapter_sha256": fingerprint(stack.modules.state_dict()),
                    "parameters": sum(p.numel() for p in stack.modules.parameters()),
                    "training_width": WIDTHS[0], "steps": STEPS, "tail_targets": TAIL, "kl_stride": KL_STRIDE}


@torch.no_grad()
def cache(root, source, work, seed):
    model, stack, _, _, checkpoint, attention = load(root, source, work, seed, SparseTrainingAttention)
    stack.backend("explicit")
    windows, keys = identity(work, seed, checkpoint, stack)
    stack.context(None, False)
    attention.mode = "flash"
    reference = []
    with attention.installed(), mixed_precision():
        for i, row in enumerate(windows):
            ids = row[None,:-1].cuda()
            phase = i % KL_STRIDE
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -TAIL:]
            reference.append(hidden[0, phase::KL_STRIDE].cpu().contiguous())
    tape = torch.stack(reference)
    assert tape.dtype == torch.bfloat16 and tuple(tape.shape) == (STEPS, TAIL//KL_STRIDE, model.config.hidden_size)
    path = work / f"sparse-teacher-{seed}.pt"
    save(path, {"identity": keys, "tape": tape})
    write_json(work / f"sparse-teacher-{seed}.json", keys | {"sha256": digest(path), "shape": list(tape.shape)})


def optimizer_state(model, stack):
    assert not any(p.requires_grad for p in model.parameters())
    params = list(stack.modules.parameters())
    return params, torch.optim.AdamW(params, lr=PEAK_LR, fused=True, weight_decay=.01)


def step_loss(model, data, tape, step):
    phase = (step-1) % KL_STRIDE
    with mixed_precision():
        hidden = model.model(input_ids=data[:, :-1], use_cache=False).last_hidden_state
        student = model.lm_head(hidden[:, -TAIL:]).float()
        with torch.no_grad():
            teacher = model.lm_head(tape[None]).float().log_softmax(-1)
    nll = F.cross_entropy(student.flatten(0,1), data[:, -TAIL:].flatten())
    kl = F.kl_div(student[:,phase::KL_STRIDE].log_softmax(-1), teacher, log_target=True, reduction="batchmean")
    return nll, kl / teacher.size(1)


def train(root, source, work, index, *, pilot=False):
    seed, arm = RUNS[index]
    model, stack, _, _, checkpoint, attention = load(root, source, work, seed, SparseTrainingAttention)
    stack.backend("explicit")
    stack.context(None, False)
    windows, expected = identity(work, seed, checkpoint, stack)
    tape_path = work / f"sparse-teacher-{seed}.pt"
    receipt = read(work / f"sparse-teacher-{seed}.json")
    tape_hash = digest(tape_path)
    assert receipt["sha256"] == tape_hash
    frozen = torch.load(tape_path, map_location="cpu", weights_only=True)
    assert frozen["identity"] == expected and all(receipt[k] == v for k,v in expected.items())
    tape = frozen["tape"].cuda()
    del frozen
    windows = windows.cuda()
    signature = expected | {"training_arm": arm, "teacher_sha256": tape_hash, "kl": KL, "peak_lr": PEAK_LR}
    name = f"{seed}-{arm}" + ("-pilot" if pilot else "")
    restart = work / f"sparse-adaptation-restart-{name}.pt"
    params, optimizer = optimizer_state(model, stack)
    start, elapsed, history = 0, 0., []
    if restart.exists():
        saved = torch.load(restart, map_location="cuda", weights_only=True)
        assert saved["identity"] == signature
        stack.modules.load_state_dict(saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        start, elapsed, history = saved["step"], saved["elapsed"], saved["history"]
        del saved

    attention.mode = arm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    nsteps = 3 if pilot else STEPS
    with attention.installed():
        for step in range(start+1, nsteps+1):
            progress = max(0, (step-WARMUP)/(STEPS-WARMUP))
            optimizer.param_groups[0]["lr"] = PEAK_LR * min(1, step/WARMUP) * (.1+.45*(1+math.cos(math.pi*progress)))
            optimizer.zero_grad(set_to_none=True)
            nll, kl = step_loss(model, windows[step-1:step], tape[step-1], step)
            obj = nll + KL * kl
            obj.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if pilot or step in (1, 8, 16, 32, 64, 96, STEPS):
                torch.cuda.synchronize()
                assert float(norm) > 0
                elapsed += perf_counter()-started
                history.append({"step": step, "seconds": elapsed, "nll": float(nll.detach()), "kl": float(kl.detach()),
                                "preclip_norm": float(norm)})
                print(name, history[-1], flush=True)
                save(restart, {"identity": signature, "adapter": stack.modules.state_dict(), "optimizer": optimizer.state_dict(),
                               "step": step, "elapsed": elapsed, "history": history})
                started = perf_counter()
    assert history[-1]["step"] == nsteps and fingerprint(stack.modules.state_dict()) != signature["initial_adapter_sha256"]
    out = work / f"sparse-adapted-{name}.pt"
    save(out, {"identity": signature, "adapter": stack.modules.state_dict(), "step": nsteps})
    write_json(work / f"sparse-train-{name}.json", signature | {"history": history, "adapter_sha256": digest(out),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "train_tokens": nsteps*WIDTHS[0]})


@torch.inference_mode()
def evaluate(root, source, work, index):
    seed, arm = RUNS[index]
    model, stack, projection, _, checkpoint, attention = load(root, source, work, seed, WindowAttention)
    name = f"{seed}-{arm}"
    path = work / f"sparse-adapted-{name}.pt"
    trained = read(work / f"sparse-train-{name}.json")
    assert digest(path) == trained["adapter_sha256"]
    trained_adapter = torch.load(path, map_location="cpu", weights_only=True)
    signature = trained_adapter["identity"]
    assert all(trained[k] == v for k,v in signature.items())
    assert (signature["source_checkpoint_sha256"], signature["seed"], signature["training_arm"], trained_adapter["step"]) == (
        checkpoint, seed, arm, STEPS)
    stack.modules.load_state_dict(trained_adapter["adapter"])
    transfer_path, hot_path = (work / f"{a}-manifest.json" for a in ("transfer","hotpot"))
    transfer, hot = map(read, (transfer_path, hot_path))
    outcomes = {}
    with attention.installed():
        for width in WIDTHS:
            a, b = work / f"transfer-{width}.json", work / f"hotpot-{width}.json"
            assert digest(a) == transfer["fixture"][str(width)]["sha256"] and digest(b) == hot["fixture"][str(width)]["sha256"]
            prose, qa = read(a), {"retrieval": read(b)}
            by_eval_mode = {}
            for policy in ARMS:
                attention.mode = policy
                by_eval_mode[policy] = {"pg19": language_score(model, stack, prose, context=width),
                       "retrieval": retrieval_score(model, stack, projection, prose, context=width),
                       "hotpot": retrieval_score(model, stack, projection, qa, context=width)}
            outcomes[width] = by_eval_mode
    write_json(work / f"sparse-eval-{name}.json", signature | {"updated_adapter_sha256": digest(path),
        "musique_manifest_sha256": digest(transfer_path), "hotpot_manifest_sha256": digest(hot_path), "outcomes": outcomes})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "source", "work"):
        cli.add_argument(key, type=Path)
    cli.add_argument("mode", choices=("cache","pilot","train","evaluate"))
    cli.add_argument("index", type=int, choices=range(len(RUNS)))
    a = cli.parse_args()
    if a.mode == "cache":
        cache(a.root, a.source, a.work, SEEDS[a.index])
    elif a.mode == "evaluate":
        evaluate(a.root, a.source, a.work, a.index)
    else:
        train(a.root, a.source, a.work, a.index, pilot=a.mode=="pilot")
