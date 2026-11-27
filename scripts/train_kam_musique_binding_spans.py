#!/usr/bin/env python3
"""Fit an extractive intermediate reader on disjoint MuSiQue evidence."""

import argparse
import gzip
from hashlib import sha256
import json
from math import cos, pi
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from audit_kam_natural_data import digest, words
from prepare_kam_musique_binding_spans import literal_spans
from run_kam_musique_bindings import BASE_REVISION

CAP = 256
PASSAGE = 128
SPAN_CAP = 24


class TokenHead(nn.Module):
    def __init__(self, assets):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(assets / "encoder", local_files_only=True,
                                                use_safetensors=True).eval()
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, ids, mask):
        return self.head(self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state).float()


def tokenize(tokenizer, questions, evidence):
    cooked = tokenizer(questions, evidence, padding="max_length", truncation="only_second", max_length=CAP,
                       return_tensors="pt", return_offsets_mapping=True)
    segments = torch.tensor([[i == 1 for i in cooked.sequence_ids(j)] for j in range(len(questions))])
    offsets = cooked["offset_mapping"]
    return cooked["input_ids"], cooked["attention_mask"], segments, offsets


def best_tokens(logits, mask):
    positions = torch.arange(logits.shape[1], device=logits.device)
    lengths = positions[None, :] - positions[:, None]
    possible = mask[:, :, None] & mask[:, None, :] & lengths.ge(0) & lengths.lt(SPAN_CAP)
    start, end = logits.unbind(-1)
    flat = (start[:, :, None] + end[:, None, :]).masked_fill(~possible, -torch.inf).flatten(1)
    score, choice = flat.max(dim=1)
    margin = score - start[:, 0] - end[:, 0]
    width = logits.shape[1]
    return torch.stack((choice // width, choice % width), dim=-1), margin


def decode_span(evidence, offsets, choice):
    start, end = choice
    return evidence[int(offsets[start, 0]):int(offsets[end, 1])].strip()


def aligned(offsets, mask, evidence, answer):
    if answer is None:
        return (0, 0)
    spans = literal_spans(evidence, answer)
    for low, high in spans:
        left = (offsets[:, 0].eq(low) & mask).nonzero().flatten().tolist()
        right = (offsets[:, 1].eq(high) & mask).nonzero().flatten().tolist()
        for start in left:
            for end in right:
                if 0 <= end-start < SPAN_CAP:
                    return (start, end)
    return None


def bank(rows, tokenizer, passage_tokenizer):
    limited = passage_tokenizer([r["evidence"] for r in rows], add_special_tokens=False, truncation=True,
                               max_length=PASSAGE)["input_ids"]
    evidence = passage_tokenizer.batch_decode(limited)
    ids, attention, context, offsets = tokenize(tokenizer, [r["question"] for r in rows], evidence)
    labels = [aligned(o, m, text, row["answer"]) for o, m, text, row in zip(offsets, context, evidence, rows, strict=True)]
    kept = []
    for i in range(0, len(rows), 2):
        pos, neg = rows[i:i+2]
        assert pos["has_answer"] and not neg["has_answer"]
        assert (pos["id"], pos["step"], pos["question"]) == (neg["id"], neg["step"], neg["question"])
        if labels[i] is not None and context[i+1].any():
            kept.extend((i, i+1))
    targets = torch.tensor([labels[i] for i in kept], dtype=torch.long, device="cuda")
    tensors = tuple(t[kept].cuda() for t in (ids, attention, context))
    meta = {"source_rows": len(rows), "eligible_rows": len(kept),
            "retained_sha256": sha256(json.dumps(kept, separators=(",", ":")).encode()).hexdigest()}
    return tensors, targets, meta, (offsets[kept], [evidence[i] for i in kept], [rows[i]["answer"] for i in kept])


@torch.inference_mode()
def monitor(model, data, details):
    tokens, _, _, (offsets, evidence, answer) = data
    right, abstain, callable_ = 0, 0, 0
    answered_correctly = 0
    for start in range(0, len(answer), 32):
        ids, mask, context = (v[start:start+32] for v in tokens)
        logit = model(ids, mask)
        selected, diff = best_tokens(logit, context)
        for j, (pair, margin) in enumerate(zip(selected.tolist(), diff.tolist(), strict=True), start):
            prediction = decode_span(evidence[j], offsets[j], pair)
            if answer[j] is None:
                abstain += margin <= 0
            else:
                found = words(prediction) == words(answer[j])
                right += found
                answered_correctly += found and margin > 0
                callable_ += margin > 0
    return {"positive": len(answer)//2, "negative": len(answer)//2,
            "positive_exact_before_abstention": right, "positive_exact_with_margin_zero": answered_correctly,
            "positive_not_abstained": callable_, "negative_abstained": abstain, "tag": details}


def fit(a):
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    data = json.loads(gzip.decompress(a.data.read_bytes()))
    tokenizer = AutoTokenizer.from_pretrained(a.assets / "encoder", local_files_only=True)
    reader = AutoTokenizer.from_pretrained(a.assets / f"reader_base_{BASE_REVISION}", local_files_only=True)
    assert tokenizer.is_fast
    fit_data, validation = [bank(data["splits"][k], tokenizer, reader) for k in ("fit", "monitor")]
    tensors, target, provenance, _ = fit_data
    model = TokenHead(a.assets).cuda().eval()
    assert all(p.device.type == "cuda" for p in model.parameters())
    frozen = a.arm == "frozen"
    for p in model.encoder.parameters():
        p.requires_grad_(not frozen)
    fitting = [{"params": list(model.head.parameters()), "lr": .0005}]
    if not frozen:
        fitting.append({"params": list(model.encoder.parameters()), "lr": .00002})
    optimizer = torch.optim.AdamW(fitting, fused=True)
    rates = [q["lr"] for q in optimizer.param_groups]
    generator = torch.Generator().manual_seed(a.seed + 101026)
    orders = torch.randint(len(target), (a.steps, a.batch), generator=generator)
    sample_sha = sha256(orders.numpy().tobytes()).hexdigest()
    indexes = orders.cuda()
    losses = []
    initial = sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.head.parameters())).hexdigest()
    torch.cuda.synchronize()
    t0 = perf_counter()
    for i, batch in enumerate(indexes, 1):
        ids, mask, context = [v[batch] for v in tensors]
        allowed = context.clone()
        allowed[:, 0] = True
        labels = target[batch]
        scale = min(i/32, 1) * (.15 + .85 * (1 + cos(pi*i/a.steps))/2)
        for rate, group in zip(rates, optimizer.param_groups, strict=True):
            group["lr"] = rate * scale
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(ids, mask)
        start, end = logit.masked_fill(~allowed.unsqueeze(-1), -torch.inf).unbind(-1)
        loss = (F.cross_entropy(start, labels[:, 0]) + F.cross_entropy(end, labels[:, 1])) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if i in (64, a.steps//2, a.steps):
            losses.append({"step": i, "loss": float(loss.detach())})
    torch.cuda.synchronize()
    elapsed = perf_counter() - t0
    evaluated = monitor(model, validation, "fp32_monitor")
    assets = json.loads((a.assets / "revisions.json").read_text())
    result = {"arm": a.arm, "assets": assets, "source_sha256": data["source_sha256"], "data_sha256": digest(a.data),
              "encoder_sha256": digest(a.assets / "encoder/model.safetensors"), "device": torch.cuda.get_device_name(),
              "steps": a.steps, "batch": a.batch, "seed": a.seed, "cap": CAP, "evidence_passage_cap": PASSAGE,
              "initial_head_sha256": initial, "sample_sha256": sample_sha, "fit": provenance,
              "monitor_records": validation[2], "elapsed_seconds": elapsed, "losses": losses, "monitor": evaluated}
    torch.save({"provenance": result, "weights": model.state_dict()}, a.weights)
    return result | {"weights_sha256": digest(a.weights)}


class SpanReader:
    def __init__(self, assets, weights, *, abstain):
        self.tokenizer = AutoTokenizer.from_pretrained(assets / "encoder", local_files_only=True)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        assert state["provenance"]["encoder_sha256"] == digest(assets / "encoder/model.safetensors")
        self.model = TokenHead(assets).cuda().eval()
        self.model.load_state_dict(state["weights"])
        self.abstain = abstain

    @torch.inference_mode()
    def answers(self, prompts, *, abstain=None):
        abstain = self.abstain if abstain is None else abstain
        extracted = []
        for i in range(0, len(prompts), 32):
            pair = [p.split(" Question: ", 1)[1].split(" Evidence: ", 1) for p in prompts[i:i+32]]
            qs, passages = map(list, zip(*pair, strict=True))
            ids, mask, context, offsets = tokenize(self.tokenizer, qs, passages)
            logits = self.model(ids.cuda(), mask.cuda())
            winners, separate = best_tokens(logits, context.cuda())
            extracted.extend(decode_span(t, o, pick) if conf > 0 or not abstain else ""
                             for t, o, pick, conf in zip(passages, offsets, winners.tolist(), separate.tolist(), strict=True))
        return extracted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "data", "weights", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--arm", choices=("frozen", "trained"), required=True)
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    a.output.write_text(json.dumps(fit(a), sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
