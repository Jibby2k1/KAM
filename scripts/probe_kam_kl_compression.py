#!/usr/bin/env python3
"""Compare exact KL with top-k teacher probabilities plus a tail bucket."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from eval_kam_adapter_preservation import load_checkpoint
from kam_pretrained_adapters import mixed_precision
from train_kam_adapter_preservation import BATCHES, DOMAINS, KL_STRIDE, LENGTH, batch, corpus, layout

TOP_K = (16, 64, 256, 1024)


def grouped_kl(logp, logq, k):
    values, indices = logp.topk(k, dim=-1)
    selected = logq.gather(-1, indices)
    teacher = values.exp()
    student = selected.exp()
    eps = torch.finfo(logp.dtype).eps
    other_p = (1 - teacher.sum(-1)).clamp_min(eps)
    other_q = (1 - student.sum(-1)).clamp_min(eps)
    head = (teacher * (values - selected)).sum(-1)
    tail = other_p * (other_p.log() - other_q.log())
    return head + tail, other_p


def vector(grads):
    return torch.cat([g.detach().reshape(-1).float() for g in grads])


def comparison(reference, approximation):
    return {"relative_l2": float((reference - approximation).norm() / reference.norm()),
            "cosine": float(F.cosine_similarity(reference, approximation, dim=0))}


def measure_batch(model, stack, frames, phase):
    params = list(stack.modules.parameters())
    x = frames[:, :-1]
    stack.active = False
    with torch.no_grad(), mixed_precision():
        teacher = model.model(input_ids=x, use_cache=False).last_hidden_state
        logp = model.lm_head(teacher[:, phase::KL_STRIDE]).float().log_softmax(-1)
    stack.active = True
    with mixed_precision():
        hidden = model.model(input_ids=x, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden)
        prediction = F.cross_entropy(logits.flatten(0, 1), frames[:, 1:].reshape(-1))
        logq = logits[:, phase::KL_STRIDE].float().log_softmax(-1)
        full = F.kl_div(logq, logp, log_target=True, reduction="none").sum(-1)
    exact = vector(torch.autograd.grad(full.mean(), params, retain_graph=True))
    exact_objective = vector(torch.autograd.grad(prediction + 4 * full.mean(), params, retain_graph=True))
    full_rows = [float(group.mean()) for group in full.detach().split(BATCHES)]
    alternatives = []
    for k in TOP_K:
        approx, tail = grouped_kl(logp, logq, k)
        assert torch.isfinite(approx).all() and bool((approx <= full + 3e-5).all())
        gradient = vector(torch.autograd.grad(approx.mean(), params, retain_graph=True))
        simulated = vector(torch.autograd.grad(prediction + 4 * approx.mean(), params, retain_graph=True))
        approximate_rows = [float(group.mean()) for group in approx.detach().split(BATCHES)]
        tail_values = [float(t.mean()) for t in tail.split(BATCHES)]
        alternatives.append({"k": k, "gradient_vs_exact_kl": comparison(exact, gradient),
                             "gradient_vs_exact_objective": comparison(exact_objective, simulated),
                             "domain_kl": dict(zip(DOMAINS, approximate_rows, strict=True)),
                             "mean_teacher_tail_mass": dict(zip(DOMAINS, tail_values, strict=True))})
    return {"phase": phase, "domain_exact_kl": dict(zip(DOMAINS, full_rows, strict=True)),
            "alternatives": alternatives}


def probe(root, arm):
    seed = 731221
    model, _, stack, metadata = load_checkpoint(root, seed, "high_lr_kl", arm)
    torch.use_deterministic_algorithms(True)
    file = root / "token-manifest.json"
    data = corpus(root, "validation", json.loads(file.read_text()))
    plan = [indices.cuda() for indices in layout(data, 4, 83011)]
    offsets = torch.arange(LENGTH + 1, device="cuda")
    rows = []
    for phase in range(KL_STRIDE):
        frames = batch(data, [source[phase] for source in plan], offsets)
        rows.append(measure_batch(model, stack, frames, phase))
        print(metadata["model"], arm, phase, flush=True)
    payload = {"checkpoint": metadata, "gpu": torch.cuda.get_device_name(),
               "vocab_size": model.config.vocab_size, "tokens_per_batch": LENGTH * sum(BATCHES),
               "bytes_per_position_exact_fp32": model.config.vocab_size * 4,
               "bytes_per_position_top_k_plus_tail": {k: 8 * k + 4 for k in TOP_K}, "batches": rows}
    output = root / f"kl-compression-{arm}.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--arm", choices=("kam", "dot"), required=True)
    args = parser.parse_args()
    probe(args.root, args.arm)
