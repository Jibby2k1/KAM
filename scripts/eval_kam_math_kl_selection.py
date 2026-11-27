#!/usr/bin/env python3
"""Run both locked answer formats on a frozen adapter checkpoint."""
import argparse
import json
from pathlib import Path
from time import monotonic

import torch

from eval_kam_adapter_preservation import load_checkpoint
from eval_kam_pretrained import CONTEXT, generate
from kam_adapter_preservation import digest, key, model_identity
from kam_math_kl_protocol import BATCH_SIZE, BUDGET, DEV_RUNS, MARKERS, PROTOCOL, patterns, prompt, score, test_runs, write_json


def evaluate(root, work, split, index):
    fixture = work / f"{split}.json"
    manifest_path = work / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"] == PROTOCOL
    assert manifest["fixtures"][split]["sha256"] == digest(fixture)
    selection_sha = None
    if split == "test":
        selection_path = work / "selection.json"
        selection = json.loads(selection_path.read_text())
        assert selection["protocol"] == PROTOCOL and selection["manifest_sha256"] == digest(manifest_path)
        assert selection["model"] == model_identity(root)[0]
        runs = test_runs(selection["selected_recipe"])
        selection_sha = digest(selection_path)
    else:
        runs = DEV_RUNS
    seed, recipe, arm = runs[index]
    ident = key(seed, recipe, arm)
    target = work / f"{split}-{ident}.json"
    model, tokenizer, stack, meta = load_checkpoint(root, seed, recipe, arm)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    rows = json.loads(fixture.read_text())
    assert [row["id"] for row in rows] == manifest["fixtures"][split]["ids"]
    encoded = {}
    for label, marker in MARKERS.items():
        texts = [
            tokenizer.apply_chat_template([{"role": "user", "content": prompt(row["problem"], marker)}],
                                          tokenize=False, add_generation_prompt=True)
            for row in rows
        ]
        encoded[label] = tokenizer(texts, add_special_tokens=False)["input_ids"]
    assert max(len(ids) for family in encoded.values() for ids in family) <= CONTEXT
    order = sorted(range(len(rows)), key=lambda i: (max(len(family[i]) for family in encoded.values()), i))
    results = [{"id": row["id"]} for row in rows]
    seconds = {}
    for label, marker in MARKERS.items():
        start_time = monotonic()
        parser, stop = patterns(marker)
        family = encoded[label]
        for start in range(0, len(order), BATCH_SIZE):
            group = order[start:start+BATCH_SIZE]
            trace = []
            generations = generate(model, stack, tokenizer, [family[i] for i in group], "gsm8k",
                                   max_new_tokens=BUDGET, trace=trace, numeric_stop_pattern=stop)
            for i, response, (tokens, reason) in zip(group, generations, trace, strict=True):
                parsed, correct = score(response, rows[i]["answer"], parser)
                results[i][label] = {"parsed": parsed, "correct": correct, "output_tokens": len(tokens),
                                     "stop": reason, "prompt_tokens": len(family[i])}
            print(split, ident, label, start + len(group), len(rows), flush=True)
        torch.cuda.synchronize()
        seconds[label] = monotonic() - start_time
    write_json(target, {"metadata": {**meta, "split": split, "gpu": gpu, "protocol": PROTOCOL,
                                      "manifest_sha256": digest(manifest_path), "selection_sha256": selection_sha,
                                      "fixture_sha256": digest(fixture), "seconds": seconds},
                        "records": results})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("work", type=Path)
    cli.add_argument("split", choices=("dev", "test"))
    cli.add_argument("index", type=int)
    args = cli.parse_args()
    evaluate(args.root, args.work, args.split, args.index)
