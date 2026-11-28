#!/usr/bin/env python3
"""Audit retained-softmax-mass certificates on pretrained queries and keys."""
import argparse
import math
from pathlib import Path

import torch

from audit_kam_omitted_attention import (
    BLOCK, BOOKS, CONFIG, CONTEXT, GROUP, REVISION, SLOTS, TILE, WEIGHTS, dataset,
    load_model, observe_dense, protected_blocks, sha, write_json,
)

PROTOCOL = "kam-pretrained-mass-certificate-v1"
OFFSETS = (0, 8, 16, 31)
THRESHOLDS = (0.90, 0.95, 0.99)


def log_remainder(bound):
    # For tiny scores the integral is bounded by exp(bound)/2, avoiding cancellation.
    x = bound.clamp_min(1e-3)
    exact = x + torch.log1p(-(1 + x) * (-x).exp()) - 2 * x.log()
    return torch.where(bound >= 1e-3, exact, bound - math.log(2))


def compact_upper(query, center, radius, gram, scaling):
    scaled = query * scaling
    score_mean = scaled @ center.transpose(1, 2)
    query_norm = scaled.norm(dim=-1, keepdim=True)
    span = query_norm * radius.unsqueeze(1)
    variance_upper = query_norm.square() * gram.unsqueeze(1)
    log_chord = torch.logaddexp(span, -span) - math.log(2)
    log_taylor = torch.logaddexp(torch.zeros_like(span), log_remainder(span) + variance_upper.log())
    return math.log(GROUP) + score_mean + torch.minimum(log_chord, log_taylor)


def score_interval_upper(logits):
    lower, upper = logits.amin(dim=-1), logits.amax(dim=-1)
    mu = logits.mean(dim=-1)
    width = upper - lower
    share = ((mu - lower) / torch.where(width > 0, width, 1.0)).clamp(0, 1)
    chord = math.log(GROUP) + torch.logaddexp(lower + torch.log1p(-share), upper + share.log())
    variance = logits.var(dim=-1, correction=0)
    taylor = math.log(GROUP) + mu + torch.logaddexp(
        torch.zeros_like(mu), log_remainder((upper - mu).clamp_min(0)) + variance.log())
    return torch.minimum(chord, taylor)


def coverage(query, key, positions, scaling):
    groups = key.unflatten(1, (-1, GROUP))
    center = groups.mean(dim=2)
    residual = groups - center.unsqueeze(2)
    radius = residual.norm(dim=-1).amax(dim=-1)
    gram = (residual @ residual.transpose(-1, -2)).abs().sum(dim=-1).amax(dim=-1) / GROUP
    max_key_norm = groups.norm(dim=-1).amax(dim=-1)
    current = positions // BLOCK
    query_rows = query[:, positions]
    representative = query[:, positions // TILE * TILE]
    picked = protected_blocks(representative, center, current)
    assert (picked.sum(dim=-1) == SLOTS).all()
    logits = (query_rows * scaling) @ key.transpose(1, 2)
    causal = torch.arange(key.size(1)) <= positions[:, None]
    retained = picked.repeat_interleave(BLOCK, dim=-1) & causal
    kept_sum = logits.masked_fill(~retained, -torch.inf).logsumexp(dim=-1)
    all_sum = logits.masked_fill(~causal, -torch.inf).logsumexp(dim=-1)
    actual = (kept_sum - all_sum).exp()
    valid = torch.arange(picked.size(-1)) < current[:, None] - 1
    omitted = (~picked & valid).repeat_interleave(BLOCK // GROUP, dim=-1)
    assert (omitted.sum(dim=-1) > 0).all()
    group_logits = logits.unflatten(-1, (-1, GROUP))
    exact_group = group_logits.logsumexp(dim=-1)
    drift = (query_rows - representative).norm(dim=-1, keepdim=True) * scaling * max_key_norm.unsqueeze(1)
    bounds = {"compact_tile": compact_upper(representative, center, radius, gram, scaling) + drift,
              "compact_query": compact_upper(query_rows, center, radius, gram, scaling),
              "exact_interval": score_interval_upper(group_logits)}
    outputs = {"actual": actual}
    for label, bound in bounds.items():
        assert ((bound - exact_group)[omitted] >= -1e-10).all(), label
        upper = bound.masked_fill(~omitted, -torch.inf).logsumexp(dim=-1)
        lower = torch.sigmoid(kept_sum - upper)
        assert (lower <= actual + 1e-10).all(), label
        outputs[label] = lower
    return outputs


def counts(mass):
    # Sample positions were constructed as consecutive four-offset groups.
    rows = mass.unflatten(-1, (-1, len(OFFSETS)))
    return {
        "samples_per_offset": rows.numel() // len(OFFSETS),
        "p50": float(mass.median()),
        "counts": {str(mark): (rows >= mark).sum(dim=(0, 1)).tolist() for mark in THRESHOLDS},
    }


def audit(model, tokens, positions):
    result = {}

    def assess(module, query, keys, _values, _dense, scaling):
        scores = coverage(query.double(), keys.double(), positions, scaling)
        result[module.layer_idx] = {variant: counts(value) for variant, value in scores.items()}

    observe_dense(model, tokens, assess, tag=PROTOCOL)
    assert len(result) == model.config.num_hidden_layers
    return [result[layer] for layer in sorted(result)]


def aggregate(books):
    rows = [layer for book in books for layer in book["layers"]]
    totals = {}
    for variant in rows[0]:
        tests = sum(row[variant]["samples_per_offset"] for row in rows)
        threshold_counts = {
            str(mark): [sum(row[variant]["counts"][str(mark)][j] for row in rows) for j in range(len(OFFSETS))]
            for mark in THRESHOLDS
        }
        totals[variant] = {"counts_by_offset": threshold_counts, "samples_per_offset": tests}
    return totals


@torch.inference_mode()
def main(args):
    torch.set_num_threads(args.threads)
    model, tokenizer = load_model(args.model)
    positions = (torch.arange(544, CONTEXT, BLOCK)[:, None] + torch.tensor(OFFSETS)).flatten()
    rows = []
    for record, tokens in dataset(tokenizer, args.sources, args.books, limit=args.limit):
        rows.append({**record, "layers": audit(model, tokens, positions)})
        print("verified certificate book", record["id"], flush=True)
    write_json(args.output, {"protocol": PROTOCOL, "revision": REVISION, "model_sha256": WEIGHTS,
                            "config_sha256": CONFIG, "positions": positions.tolist(), "offsets": OFFSETS,
                            "source_list_sha256": sha(args.sources.read_bytes()), "torch": torch.__version__,
                            "books": rows, "totals": aggregate(rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "sources", "books", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--limit", type=int, default=BOOKS, choices=(1, BOOKS))
    parser.add_argument("--threads", type=int, default=4)
    main(parser.parse_args())
