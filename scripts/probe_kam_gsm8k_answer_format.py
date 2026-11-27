#!/usr/bin/env python3
"""Compare a new answer marker on the original frozen GSM8K questions."""
import argparse
import json
from pathlib import Path

import torch

from eval_kam_adapter_preservation import load_checkpoint
from eval_kam_pretrained import CONTEXT, LIMIT, generate, load_cases, numerical_answer
from eval_kam_svamp_transfer import ANSWER, ANSWER_STOP
from kam_adapter_preservation import FACTORIAL_RUNS, LLAMA_ID, LOW_RATE_KL_RECIPE, LOW_RATE_KL_RUNS, digest, key

BATCH = 64


def probe(root, index, pilot):
    seed, recipe, arm = (LOW_RATE_KL_RUNS if pilot else FACTORIAL_RUNS)[index]
    assert recipe in ("untouched", "lower_lr", LOW_RATE_KL_RECIPE)
    name = key(seed, recipe, arm)
    original = json.loads((root / f"gsm8k-{name}.json").read_text())
    model, tokenizer, stack, meta = load_checkpoint(root, seed, recipe, arm)
    assert meta["model"] == LLAMA_ID and meta["adapter_sha256"] == original["metadata"]["adapter_sha256"]
    gpu = torch.cuda.get_device_name()
    assert gpu == original["metadata"]["gpu"]
    source, cases = load_cases(root / "data", "gsm8k")
    assert digest(source) == original["metadata"]["source_sha256"]
    prompts = [f"{row['question']}\nSolve the problem. End your answer on a new line using "
               "'Answer: ' followed by the final number." for row in cases]
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                          add_generation_prompt=True) for prompt in prompts]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    assert all(len(ids) <= CONTEXT for ids in encoded)
    order = sorted(range(len(cases)), key=lambda i: (original["records"][i]["prompt_tokens"], i))
    outcomes = {}
    for start in range(0, len(order), BATCH):
        group = order[start:start+BATCH]
        generated = generate(model, stack, tokenizer, [encoded[i] for i in group], "gsm8k",
                             numeric_stop_pattern=ANSWER_STOP)
        for i, response in zip(group, generated, strict=True):
            valid = ANSWER.findall(response)
            alternate = valid[-1].replace(",", "") if valid else None
            legacy = numerical_answer(response)
            output = alternate if alternate is not None else legacy
            gold = numerical_answer(cases[i]["answer"])
            assert gold is not None
            outcomes[i] = {"index": i, "prompt_tokens": len(encoded[i]), "correct": alternate == gold,
                           "parsed": alternate is not None, "legacy_marker_parsed": legacy is not None,
                           "any_marker_correct": output == gold, "any_marker_parsed": output is not None}
    output = root / f"gsm8k-answer-format-{name}.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"metadata": {**meta, "source_sha256": digest(source), "gpu": gpu,
                                                   "generation_budget": LIMIT["gsm8k"],
                                                   "original_prompt_order": True},
                                     "records": [outcomes[i] for i in range(len(cases))]},
                                    separators=(",", ":")))
    temporary.replace(output)
    print(name, "strict", sum(r["correct"] for r in outcomes.values()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    probe(args.root, args.index, args.pilot)
