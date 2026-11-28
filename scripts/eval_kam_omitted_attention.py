#!/usr/bin/env python3
"""Compare fixed omitted-block summaries in the live SmolLM2 backbone."""
import argparse
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.llama.modeling_llama import repeat_kv

from audit_kam_omitted_attention import (
    BLOCK, BOOKS, CONFIG, CONTEXT, GROUP, REVISION, SLOTS, TILE, TOKENIZER, WEIGHTS, dataset, load_model,
    local_outputs, means, mixing_weight, protected_blocks, sha, write_json,
)

PROTOCOL = "kam-live-omitted-summary-v1"
POLICIES = ("dense", "protected", "mean16", "diagonal16", "mean8")
SCORE = 512
WARM_PREFIX = SLOTS * BLOCK
BATCHED_TILES = 8


def tile_book(query, key, value, scaling, policy):
    """Exact retained-key read, with optional omitted group contribution."""
    heads, length, dim = query.shape
    device = query.device
    assert length % BLOCK == 0 and length >= WARM_PREFIX
    dense_prefix = F.scaled_dot_product_attention(
        query[:, :WARM_PREFIX], key[:, :WARM_PREFIX], value[:, :WARM_PREFIX], scale=scaling, is_causal=True)
    tiles = query[:, WARM_PREFIX:].unflatten(1, (-1, TILE))
    starts = torch.arange(WARM_PREFIX, length, TILE, device=device)
    current = starts // BLOCK
    key_means = means(key, GROUP)
    route = protected_blocks(tiles[:, :, 0], key_means, current)
    block_ids = torch.arange(length // BLOCK, device=device)
    chosen_blocks = torch.where(route, block_ids, length // BLOCK).sort(dim=-1).values[..., :SLOTS]
    offsets = torch.arange(BLOCK, device=device)
    gather = (chosen_blocks[..., None] * BLOCK + offsets).flatten(-2)
    head_ids = torch.arange(heads, device=device)[:, None, None]
    rows = torch.arange(TILE, device=device)
    output = [dense_prefix]
    if policy != "protected":
        width = GROUP // 2 if policy == "mean8" else GROUP
        summary_key = means(key, width).transpose(1, 2).unsqueeze(1)
        summary_value = means(value, width).unsqueeze(1)
        eligible = block_ids < current[:, None] - 1
        omitted = (~route & eligible).repeat_interleave(BLOCK // width, dim=-1).unsqueeze(-2)
        if policy == "diagonal16":
            residual = key.unflatten(1, (-1, GROUP)) - key_means.unsqueeze(2)
            variance = residual.square().mean(dim=2).transpose(1, 2).unsqueeze(1)
    for start in range(0, len(starts), BATCHED_TILES):
        tile = slice(start, start + BATCHED_TILES)
        sampled = gather[:, tile]
        exact_key = key[head_ids, sampled]
        exact_value = value[head_ids, sampled]
        before = sampled.unsqueeze(-2) <= (starts[tile, None] + rows).unsqueeze(-1)
        scaled = tiles[:, tile] * scaling
        kept = (scaled @ exact_key.transpose(-1, -2)).masked_fill(~before, -torch.inf)
        exact = kept.softmax(dim=-1) @ exact_value
        if policy == "protected":
            output.append(exact.flatten(1, 2))
            continue
        log_summary = math.log(width) + (scaled @ summary_key)
        if policy == "diagonal16":
            log_summary += torch.log1p((scaled.square() @ variance) / 2)
        log_summary = log_summary.masked_fill(~omitted[:, tile], -torch.inf)
        weight = mixing_weight(kept.logsumexp(dim=-1, keepdim=True), log_summary)
        approx = weight[..., :1] * exact + weight[..., 1:] @ summary_value
        output.append(approx.flatten(1, 2))
    return torch.cat(output, dim=1)


def fixture_gate(device):
    generator = torch.Generator().manual_seed(479246)
    shape = (2, 640, 16)
    query, key, value = (torch.randn(shape, generator=generator).to(device) for _ in range(3))
    positions = torch.arange(512, 640)
    _, reference, _, _ = local_outputs(query.cpu(), key.cpu(), value.cpu(), positions, 0.25)
    for policy in POLICIES[1:]:
        got = tile_book(query, key, value, 0.25, policy).cpu()
        prefix = F.scaled_dot_product_attention(
            query.cpu()[:, :512], key.cpu()[:, :512], value.cpu()[:, :512], scale=0.25, is_causal=True)
        assert torch.allclose(got[:, :512], prefix, atol=3e-5, rtol=3e-5)
        assert torch.allclose(got[:, positions], reference[policy], atol=3e-5, rtol=3e-5), policy


def scored(model, tokens, policy):
    def model_attention(module, query, key, value, mask, scaling, **kwargs):
        if policy == "dense":
            return sdpa_attention_forward(module, query, key, value, mask, scaling=scaling, **kwargs)
        assert query.size(0) == 1 and mask is None
        keys = repeat_kv(key, module.num_key_value_groups)[0]
        values = repeat_kv(value, module.num_key_value_groups)[0]
        output = tile_book(query[0], keys, values, scaling, policy)
        return output.transpose(0, 1).unsqueeze(0).contiguous(), None

    AttentionInterface.register(PROTOCOL, model_attention)
    model.config._attn_implementation = PROTOCOL
    return token_loss(model, tokens)


def token_loss(model, tokens):
    states = model.model(tokens, use_cache=False).last_hidden_state[0, -SCORE-1:-1]
    target = tokens[0, -SCORE:]
    total = sum(F.cross_entropy(model.lm_head(x), labels, reduction="sum")
                for x, labels in zip(states.split(128), target.split(128), strict=True))
    return float(total)


def aggregate(rows):
    dense = np.array([row["nll_sum"]["dense"] / SCORE for row in rows])
    protected = np.array([row["nll_sum"]["protected"] / SCORE for row in rows])
    draw = np.random.default_rng(456110).integers(len(rows), size=(20_000, len(rows)))
    output = {}
    for policy in POLICIES:
        nll = np.array([row["nll_sum"][policy] / SCORE for row in rows])
        output[policy] = {"nll": float(nll.mean()), "ppl": math.exp(float(nll.mean())),
                          "delta_vs_dense": float((nll - dense).mean()),
                          "delta_vs_protected": float((nll - protected).mean()),
                          "paired_book_bootstrap95_vs_protected":
                              np.quantile((nll - protected)[draw].mean(axis=1), (.025, .975)).tolist(),
                          "books_beating_protected": int((nll < protected).sum())}
    return output


@torch.inference_mode()
def main(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    fixture_gate(device)
    model, tokenizer = load_model(args.model)
    model.to(device)
    inputs = dataset(tokenizer, args.sources, args.books, limit=args.limit)
    rows = []
    for index, (record, tokens) in enumerate(inputs):
        shifted = index % len(POLICIES)
        order = POLICIES[shifted:] + POLICIES[:shifted]
        input_ids = torch.tensor([tokens], device=device)
        values = {}
        for policy in order:
            start = perf_counter()
            values[policy] = scored(model, input_ids, policy)
            print(record["id"], policy, round(values[policy] / SCORE, 5),
                  f"({perf_counter() - start:.2f} s)", flush=True)
        rows.append({**record, "policy_order": order, "nll_sum": values})
        summary = {"protocol": PROTOCOL, "device": str(device), "torch": torch.__version__, "revision": REVISION,
                   "source_list_sha256": sha(args.sources.read_bytes()), "model_sha256": WEIGHTS,
                   "tokenizer_sha256": TOKENIZER, "config_sha256": CONFIG, "total_context": CONTEXT,
                   "scored_tokens_per_book": SCORE, "complete": len(rows) == args.limit, "books": rows,
                   "policies": aggregate(rows)}
        write_json(args.output.with_suffix(".partial"), summary)
    args.output.with_suffix(".partial").replace(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "sources", "books", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--limit", type=int, default=BOOKS, choices=(1, BOOKS))
    parser.add_argument("--threads", type=int, default=4)
    main(parser.parse_args())
