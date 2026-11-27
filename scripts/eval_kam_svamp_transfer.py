#!/usr/bin/env python3
"""Test frozen Llama low-rate adapters on the official SVAMP transfer set."""
import argparse
import json
import re
from decimal import Decimal
from pathlib import Path

import torch

from eval_kam_adapter_preservation import load_checkpoint
from eval_kam_pretrained import LIMIT, generate, numerical_answer, question
from kam_adapter_preservation import FACTORIAL_RUNS, LLAMA_ID, LOW_RATE_KL_RECIPE, LOW_RATE_KL_RUNS, digest, key

SVAMP_REVISION = "78e727689e1c1bebfc4be39c446898e8e10b0518"
SVAMP_SHA256 = "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"
BATCH_SIZE = 64
ANSWER = re.compile(r"^\s*Answer:\s*(-?\d[\d,]*(?:\.\d+)?)[ \t]*[.!]?[ \t]*(?:\n|\Z)", re.IGNORECASE | re.MULTILINE)
ANSWER_STOP = re.compile(r"^\s*Answer:\s*-?\d[\d,]*(?:\.\d+)?[ \t]*[.!]?[ \t]*\n", re.IGNORECASE | re.MULTILINE)


def evaluate(root, index, pilot, alternate_format):
    seed, recipe, arm = (LOW_RATE_KL_RUNS if pilot else FACTORIAL_RUNS)[index]
    assert recipe in ("lower_lr", LOW_RATE_KL_RECIPE, "untouched")
    name = key(seed, recipe, arm)
    source = root / "SVAMP.json"
    assert digest(source) == SVAMP_SHA256
    data = json.loads(source.read_text(), parse_float=Decimal)
    assert len(data) == len({row["ID"] for row in data}) == 1000
    assert all(Decimal(row["Answer"]) == int(row["Answer"]) for row in data)
    model, tokenizer, stack, metadata = load_checkpoint(root, seed, recipe, arm)
    assert metadata["model"] == LLAMA_ID
    problems = [row["Body"].strip() + " " + row["Question"].strip() for row in data]
    if alternate_format:
        prompts = [p + "\nSolve the problem. End your answer on a new line using "
                       "'Answer: ' followed by the final number." for p in problems]
    else:
        prompts = [question("gsm8k", {"question": problem}, i)[0] for i, problem in enumerate(problems)]
    text = [tokenizer.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                          add_generation_prompt=True) for msg in prompts]
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    lengths = [len(row) for row in encoded]
    order = sorted(range(len(encoded)), key=lambda i: (lengths[i], i))
    results = {}
    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start:start+BATCH_SIZE]
        pattern = ANSWER_STOP if alternate_format else None
        generated = generate(model, stack, tokenizer, [encoded[i] for i in indices], "gsm8k",
                             numeric_stop_pattern=pattern)
        for i, completion in zip(indices, generated, strict=True):
            legacy = numerical_answer(completion)
            matches = ANSWER.findall(completion) if alternate_format else []
            answer = (matches[-1].replace(",", "") if matches else None) if alternate_format else legacy
            gold = Decimal(data[i]["Answer"])
            kind = data[i]["Type"].replace("Common-Divison", "Common-Division")
            results[i] = {"index": i, "id": data[i]["ID"], "type": kind, "prompt_tokens": lengths[i],
                          "parsed": answer is not None, "correct": answer is not None and Decimal(answer) == gold}
            if alternate_format:
                either = answer if answer is not None else legacy
                results[i].update(legacy_marker_parsed=legacy is not None, any_marker_parsed=either is not None,
                                  any_marker_correct=either is not None and Decimal(either) == gold)
        if start % (BATCH_SIZE * 5) == 0:
            print(name, start + len(indices), len(encoded), flush=True)
    label = "svamp-answer-format" if alternate_format else "svamp"
    target = root / f"{label}-{name}.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps({"metadata": {**metadata, "source_revision": SVAMP_REVISION,
                                             "source_sha256": SVAMP_SHA256, "generation_budget": LIMIT["gsm8k"],
                                             "gpu": torch.cuda.get_device_name()},
                                "records": [results[i] for i in range(len(data))]}, separators=(",", ":")))
    temp.replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--alternate-format", action="store_true")
    args = parser.parse_args()
    evaluate(args.root, args.index, args.pilot, args.alternate_format)
