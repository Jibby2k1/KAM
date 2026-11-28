#!/usr/bin/env python3
"""Replay one original bucket at 384 tokens and check the content-stop bypass."""
import argparse
import json
from pathlib import Path
import re
from time import monotonic

import torch

from eval_kam_adapter_preservation import load_checkpoint
from eval_kam_full_math500 import PREFIX_BUDGET, PROMPT_SUFFIX, SELECTIONS
from eval_kam_pretrained import generate
from kam_adapter_preservation import digest, key, model_identity
from kam_math_kl_protocol import BATCH_SIZE, SOURCES, write_json


def load_panel(root, work, arm):
    model_id = model_identity(root)[0]
    recipe, _ = SELECTIONS[model_id]
    name = key(731221, recipe, arm)
    raw_path = work / f"neutral-{name}.json"
    original = json.loads(raw_path.read_text())
    source = work / "math500-test.jsonl"
    assert digest(source) == SOURCES[source.name]
    data = [json.loads(line) for line in source.read_text().splitlines()]
    model, tokenizer, stack, metadata = load_checkpoint(root, 731221, recipe, arm)
    assert original["metadata"]["adapter_sha256"] == metadata["adapter_sha256"]
    assert original["metadata"]["gpu"] == torch.cuda.get_device_name()
    records = original["records"]
    assert [row["unique_id"] for row in data] == [row["id"] for row in records]
    order = sorted(range(len(data)), key=lambda i: (records[i]["prompt_tokens"], i))
    start = 2 * BATCH_SIZE
    group = order[start:start+BATCH_SIZE]
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": data[i]["problem"] + PROMPT_SUFFIX}],
                                      tokenize=False, add_generation_prompt=True)
        for i in group
    ]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    assert [len(row) for row in encoded] == [records[i]["prompt_tokens"] for i in group]
    reference = [(records[i]["prefix"] if records[i]["prefix"] is not None else records[i]["completion"],
                  min(PREFIX_BUDGET, records[i]["output_tokens"])) for i in group]
    provenance = {
        "model": model_id, "arm": arm, "seed": 731221, "gpu": torch.cuda.get_device_name(),
        "source_sha256": digest(source), "adapter_sha256": metadata["adapter_sha256"],
        "generation_sha256": digest(raw_path), "n": len(group),
        "rows": [{"id": records[i]["id"], "prefix_tokens": min(PREFIX_BUDGET, records[i]["output_tokens"])}
                 for i in group],
    }
    return name, model, tokenizer, stack, encoded, reference, provenance


def probe(root, work, arm):
    name, model, tokenizer, stack, encoded, reference, provenance = load_panel(root, work, arm)
    timings = {"bypass": [], "never-match": []}
    impossible = re.compile(r"(?!)")
    torch.cuda.synchronize()
    for mode in ("bypass", "never-match", "never-match", "bypass"):
        trace = []
        start_time = monotonic()
        outputs = generate(model, stack, tokenizer, encoded, "gsm8k", max_new_tokens=PREFIX_BUDGET,
                           trace=trace, numeric_stop_pattern=impossible, stop_on_content=mode == "never-match")
        torch.cuda.synchronize()
        timings[mode].append(monotonic() - start_time)
        measured = [(text, len(tokens)) for text, (tokens, _) in zip(outputs, trace, strict=True)]
        assert measured == reference, mode
    write_json(work / f"prefix-parity-{name}.json", {**provenance, "timings_s": timings})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("work", type=Path)
    cli.add_argument("arm", choices=("kam", "dot"))
    args = cli.parse_args()
    probe(args.root, args.work, args.arm)
