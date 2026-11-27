#!/usr/bin/env python3
"""Test whether a longer continuation recovers otherwise missing GSM8K answers."""
import argparse
import json
from pathlib import Path

from eval_kam_adapter_preservation import load_checkpoint
from eval_kam_pretrained import LIMIT, generate, load_cases, numerical_answer, question
from kam_adapter_preservation import FACTORIAL_RUNS, digest, key

TASK = "gsm8k"
SHORT = LIMIT[TASK]
LONG = 2 * SHORT
BATCH = 64


def probe(root, index):
    seed, recipe, arm = FACTORIAL_RUNS[index]
    assert recipe in ("lower_lr", "untouched")
    ident = key(seed, recipe, arm)
    official = json.loads((root / f"{TASK}-{ident}.json").read_text())
    model, tokenizer, stack, checkpoint = load_checkpoint(root, seed, recipe, arm)
    source, cases = load_cases(root / "data", TASK)
    assert digest(source) == official["metadata"]["source_sha256"]
    assert checkpoint["adapter_sha256"] == official["metadata"]["adapter_sha256"]
    prompts = [question(TASK, case, i)[0] for i, case in enumerate(cases)]
    text = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                          add_generation_prompt=True) for prompt in prompts]
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    lengths = [len(seq) for seq in encoded]
    assert lengths == [r["prompt_tokens"] for r in official["records"]]
    order = sorted(range(len(encoded)), key=lambda i: (lengths[i], i))
    records = {}
    for start in range(0, len(order), BATCH):
        ids = order[start:start+BATCH]
        trace = []
        outputs = generate(model, stack, tokenizer, [encoded[i] for i in ids], TASK,
                           max_new_tokens=LONG, trace=trace)
        prefixes = tokenizer.batch_decode([tokens[:SHORT] for tokens, _ in trace],
                                           skip_special_tokens=True)
        for i, head, tail, (tokens, stop) in zip(ids, prefixes, outputs, trace, strict=True):
            gold = numerical_answer(cases[i]["answer"])
            assert gold is not None
            short, long = numerical_answer(head), numerical_answer(tail)
            observed = official["records"][i]
            assert observed["index"] == i
            records[i] = {"index": i, "short_correct": short == gold, "short_parsed": short is not None,
                          "long_correct": long == gold, "long_parsed": long is not None,
                          "replay_agrees": (short == gold, short is not None) ==
                                           (observed["correct"], observed["parsed"]),
                          "output_tokens": len(tokens), "long_stop": stop}
        if start % (BATCH * 5) == 0:
            print(ident, start + len(ids), len(encoded), flush=True)
    output = root / f"gsm8k-budget-{ident}.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps({"checkpoint": checkpoint, "source_sha256": digest(source),
                                "short_budget": SHORT, "long_budget": LONG,
                                "records": [records[i] for i in range(len(cases))]},
                               separators=(",", ":")))
    temp.replace(output)
    print(ident, "short replay mismatches", sum(not row["replay_agrees"] for row in records.values()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    probe(args.root, args.index)
