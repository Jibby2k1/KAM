#!/usr/bin/env python3
"""Generate neutral-prompt MATH-500 continuations for frozen matched adapters."""
import argparse
import json
from pathlib import Path
from time import monotonic

import torch

from eval_kam_adapter_preservation import load_checkpoint
from eval_kam_pretrained import CONTEXT, generate
from kam_adapter_preservation import LLAMA_ID, MODEL_ID, digest, key, model_identity
from kam_math_kl_protocol import BATCH_SIZE, SOURCES, test_runs, write_json

PROTOCOL = "kam-math500-neutral-v1"
BUDGET = 1024
PREFIX_BUDGET = 384
PROMPT_SUFFIX = "\n\nSolve the problem. Explain your reasoning and state your final answer clearly."
SELECTIONS = {MODEL_ID: ("lower_lr_kl", 4.0), LLAMA_ID: ("lower_lr_kl05", 0.5)}


def generate_checkpoint(root, work, index):
    source = work / "math500-test.jsonl"
    source_sha = digest(source)
    assert source_sha == SOURCES[source.name]
    selection_path = work / "selection.json"
    selection = json.loads(selection_path.read_text())
    model_id = model_identity(root)[0]
    assert selection["model"] == model_id
    recipe, weight = SELECTIONS[model_id]
    assert (selection["selected_recipe"], selection["selected_kl_weight"]) == (recipe, weight)
    task = test_runs(recipe)[index]
    ident = key(*task)
    target = work / f"neutral-{ident}.json"
    model, tokenizer, stack, metadata = load_checkpoint(root, *task)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    data = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(data) == len({row["unique_id"] for row in data}) == 500
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"] + PROMPT_SUFFIX}],
            tokenize=False, add_generation_prompt=True,
        )
        for row in data
    ]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    lengths = [len(ids) for ids in encoded]
    assert max(lengths) + BUDGET <= CONTEXT
    order = sorted(range(len(data)), key=lambda i: (lengths[i], i))
    rows = {}
    start_time = monotonic()
    for start in range(0, len(order), BATCH_SIZE):
        group = order[start:start+BATCH_SIZE]
        trace = []
        texts = generate(model, stack, tokenizer, [encoded[i] for i in group], "gsm8k",
                         max_new_tokens=BUDGET, trace=trace, stop_on_content=False)
        for i, text, (tokens, stop) in zip(group, texts, trace, strict=True):
            prefix = None
            if len(tokens) > PREFIX_BUDGET:
                prefix = tokenizer.decode(tokens[:PREFIX_BUDGET], skip_special_tokens=True)
            rows[i] = {"id": data[i]["unique_id"], "prompt_tokens": lengths[i], "output_tokens": len(tokens),
                       "stop": stop, "completion": text, "prefix": prefix}
        print(ident, start + len(group), len(encoded), flush=True)
    torch.cuda.synchronize()
    write_json(target, {"metadata": {**metadata, "protocol": PROTOCOL, "source_sha256": source_sha,
                                     "selection_sha256": digest(selection_path), "gpu": gpu, "budget": BUDGET,
                                     "prefix_budget": PREFIX_BUDGET, "batch_size": BATCH_SIZE,
                                     "prompt_suffix": PROMPT_SUFFIX, "seconds": monotonic() - start_time},
                        "records": [rows[i] for i in range(len(data))]})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("work", type=Path)
    cli.add_argument("index", type=int)
    args = cli.parse_args()
    generate_checkpoint(args.root, args.work, args.index)
