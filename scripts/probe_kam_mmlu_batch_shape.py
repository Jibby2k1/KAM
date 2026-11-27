#!/usr/bin/env python3
"""Trace MMLU answer changes under production and subset batch packing."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from eval_kam_adapter_preservation import load_checkpoint
from eval_kam_pretrained import LETTERS, load_cases, pad, question
from kam_adapter_preservation import digest, key
from kam_pretrained_adapters import mixed_precision
from probe_kam_adapter_strength import fixtures

STUDY_RUNS = [(0, "untouched", "pretrained"), (731221, "high_lr_kl", "kam"), (731221, "high_lr_kl", "dot")]


@torch.inference_mode()
def scores(model, stack, tokenizer, token_rows, selected, choice_ids):
    ordered = sorted(range(len(token_rows)), key=lambda i: (len(token_rows[i]), i))
    outputs = {}
    for start in range(0, len(ordered), 64):
        group = ordered[start:start+64]
        members = [(offset, i) for offset, i in enumerate(group) if i in selected]
        if not members:
            continue
        tokens, mask = pad([token_rows[i] for i in group], tokenizer.pad_token_id)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        stack.context(mask, False)
        with mixed_precision():
            layer = model.model(input_ids=tokens, attention_mask=mask, position_ids=positions,
                                use_cache=False).last_hidden_state[:, -1:, :]
            production_head = model.lm_head(layer)[:, 0, choice_ids].float()
        float_head = F.linear(layer[:, 0, :].float(), model.lm_head.weight[choice_ids].float())
        offsets, indices = zip(*members, strict=True)
        for i, p, f in zip(indices, production_head[list(offsets)].tolist(),
                           float_head[list(offsets)].tolist(), strict=True):
            outputs[i] = {"native_scores": p, "float32_scores": f}
    assert outputs.keys() == selected
    stack.reset()
    return outputs


def probe(root, index):
    seed, recipe, arm = STUDY_RUNS[index]
    name = key(seed, recipe, arm)
    model, tokenizer, stack, meta = load_checkpoint(root, seed, recipe, arm)
    torch.set_float32_matmul_precision("highest")
    panel = fixtures(root, tokenizer)["mmlu"]
    source, cases = load_cases(root / "data", "mmlu")
    official = json.loads((root / f"mmlu-{name}.json").read_text())
    assert digest(source) == official["metadata"]["source_sha256"] == panel["source_sha256"]
    assert meta["adapter_sha256"] == official["metadata"]["adapter_sha256"]
    assert torch.cuda.get_device_name() == official["metadata"]["gpu"]
    ids = panel["ids"]
    prompts = [question("mmlu", row, i)[0] for i, row in enumerate(cases)]
    text = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True) for p in prompts]
    full = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert [len(row) for row in full] == [r["prompt_tokens"] for r in official["records"]]
    assert [full[i] for i in ids] == panel["encoded"]
    choices = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    assert all(len(c) == 1 for c in choices)
    choice_ids = [c[0] for c in choices]
    production = scores(model, stack, tokenizer, full, set(ids), choice_ids)
    subset = scores(model, stack, tokenizer, panel["encoded"], set(range(len(ids))), choice_ids)
    rows = [{"index": i, "gold": panel["gold"][j], "original_choice": official["records"][i]["choice"],
             "production": production[i], "repacked": subset[j]} for j, i in enumerate(ids)]
    out = root / f"mmlu-batch-shape-{name}.json"
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps({"metadata": meta, "gemm": "float32 highest",
                                "gpu": torch.cuda.get_device_name(), "records": rows},
                               separators=(",", ":")))
    temp.replace(out)
    print(name, len(ids), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int, choices=range(len(STUDY_RUNS)))
    args = parser.parse_args()
    probe(args.root, args.index)
