#!/usr/bin/env python3
"""Replay locked dense-head schedules on the held-out PG-19 books."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.llama.modeling_llama import repeat_kv

from audit_kam_head_fallback import BUDGET, HEADS, LAYERS, PROTOCOL as SELECTION
from audit_kam_omitted_attention import BOOKS, CONTEXT, REVISION, WEIGHTS, dataset, load_model, sha, write_json
from eval_kam_omitted_attention import SCORE, tile_book, token_loss

PROTOCOL = "kam-live-head-fallback-v1"
RUNS = 9


def install(model, layout, device):
    all_heads = set(range(HEADS))
    sizes = [len(heads) for heads in layout]
    selections = [(torch.tensor(heads, device=device, dtype=torch.long),
                   torch.tensor(sorted(all_heads - set(heads)), device=device, dtype=torch.long))
                  for heads in layout]

    def mixed_attention(module, query, key, value, mask, scaling, **kwargs):
        count = sizes[module.layer_idx]
        if count == HEADS:
            return sdpa_attention_forward(module, query, key, value, mask, scaling=scaling, **kwargs)
        assert query.size(0) == 1 and mask is None
        keys = repeat_kv(key, module.num_key_value_groups)[0]
        values = repeat_kv(value, module.num_key_value_groups)[0]
        if count == 0:
            output = tile_book(query[0], keys, values, scaling, "protected")
        else:
            dense, sparse = selections[module.layer_idx]
            exact = F.scaled_dot_product_attention(
                query[0, dense], keys[dense], values[dense], scale=scaling, is_causal=True)
            approximate = tile_book(query[0, sparse], keys[sparse], values[sparse], scaling, "protected")
            output = torch.empty_like(query[0])
            output.index_copy_(0, dense, exact)
            output.index_copy_(0, sparse, approximate)
        return output.transpose(0, 1).unsqueeze(0).contiguous(), None

    AttentionInterface.register(PROTOCOL, mixed_attention)
    model.config._attn_implementation = PROTOCOL


def paired_scores(rows, policies):
    reference = np.array([row["nll_sum"]["whole_layers"] / SCORE for row in rows])
    draw = np.random.default_rng(993402).integers(len(rows), size=(20_000, len(rows)))
    values = {}
    for policy in policies:
        nll = np.array([row["nll_sum"][policy] / SCORE for row in rows])
        delta = nll - reference
        values[policy] = {"nll": float(nll.mean()), "perplexity": math.exp(float(nll.mean())),
                          "paired_nll_vs_whole_layer": float(delta.mean()),
                          "paired_book_bootstrap95": np.quantile(delta[draw].mean(1), (.025, .975)).tolist()}
    return values


def profile(model, tokens, layouts, device):
    policies = tuple(layouts)

    def run():
        hidden = model.model(tokens, use_cache=False).last_hidden_state[:, -1:]
        output = model.lm_head(hidden)
        torch.mps.synchronize()
        return output

    for layout in layouts.values():
        install(model, layout, device)
        for _ in range(2):
            run()
    samples = {name: [] for name in layouts}
    for step in range(RUNS):
        order = step % len(policies)
        for name in policies[order:] + policies[:order]:
            install(model, layouts[name], device)
            torch.mps.synchronize()
            start = perf_counter()
            run()
            samples[name].append((perf_counter() - start) * 1000)
    return {name: {"ms": values, "median_ms": float(np.median(values)),
                   "p95_ms": float(np.quantile(values, .95))} for name, values in samples.items()}


@torch.inference_mode()
def main(args):
    torch.set_num_threads(3)
    device = torch.device("mps")
    frozen = args.selection.read_bytes()
    selection = json.loads(frozen)
    assert selection["protocol"] == SELECTION
    candidates = ("whole_layers", "groups_up_to_4", "groups_all_layers")
    layouts = {"dense": [list(range(HEADS))] * LAYERS, "protected": [[] for _ in range(LAYERS)]}
    for policy in candidates:
        layout = selection["schedules"][policy]["dense_heads"]
        assert len(layout) == LAYERS and sum(map(len, layout)) == BUDGET
        assert all(sorted(set(heads)) == heads and set(heads) <= set(range(HEADS)) for heads in layout)
        layouts[policy] = layout
    model, tokenizer = load_model(args.model)
    assert model.config.num_attention_heads == HEADS and model.config.num_hidden_layers == LAYERS
    model.to(device)
    eligible = set(selection["test_books"])
    fixture = [(record, tokens) for record, tokens in dataset(tokenizer, args.sources, args.books, limit=BOOKS)
               if record["id"] in eligible]
    assert len(fixture) == BOOKS // 2
    rows = []
    policies = tuple(layouts)
    for i, (record, tokens) in enumerate(fixture):
        if args.smoke and i > 0:
            break
        tensor = torch.tensor([tokens], device=device)
        shift = i % len(policies)
        loss = {}
        for policy in policies[shift:] + policies[:shift]:
            start = perf_counter()
            install(model, layouts[policy], device)
            loss[policy] = token_loss(model, tensor)
            print(record["id"], policy, round(loss[policy] / SCORE, 5), f"({perf_counter()-start:.2f} s)", flush=True)
        rows.append({**record, "nll_sum": loss})
    timing = None
    if not args.smoke:
        timing = profile(model, torch.tensor([fixture[0][1]], device=device), layouts, device)
    write_json(args.output, {"protocol": PROTOCOL, "torch": torch.__version__, "device": str(device),
                            "model_sha256": WEIGHTS, "revision": REVISION, "selection_sha256": sha(frozen),
                            "source_list_sha256": sha(args.sources.read_bytes()), "books": rows,
                            "context": CONTEXT, "scored_tokens_per_book": SCORE, "timing": timing,
                            "timing_book_id": fixture[0][0]["id"] if timing else None,
                            "metrics": paired_scores(rows, policies)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "sources", "books", "selection", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
