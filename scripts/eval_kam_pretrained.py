#!/usr/bin/env python3
"""Evaluate the frozen backbone and both trained residual adapters."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from kam_pretrained_adapters import ARMS, MODEL_ID, REVISION, AdapterStack, load_model, mixed_precision
from run_kam_general_benchmarks import EXPECTED, LETTERS, load_cases, make_prompt, numerical_answer

CONTEXT = 4096
LIMIT = {"humaneval": 256, "mbpp": 256, "gsm8k": 384}
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.IGNORECASE | re.DOTALL)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def question(task, case, index):
    raw, gold = make_prompt(task, case, index)
    if task == "humaneval":
        return ("Complete the following Python program. Return the full runnable code, including "
                "the original imports and function definition, in one Python code block.\n\n"
                f"```python\n{raw}\n```", None)
    if task == "mbpp":
        return ("Write a Python function for the task. Return only a Python code block. "
                f"Use the function name and signature in this example.\n\n{case['text']}\n"
                f"Example test: {case['test_list'][0]}", None)
    if task == "gsm8k":
        return (f"{case['question']}\nSolve the problem. End your answer on a new line using "
                "'#### ' followed by the final number.", None)
    return (raw + "\nChoose exactly one of A, B, C, or D. Reply with the single letter.", gold)


@torch.inference_mode()
def next_logits(model, stack, tokens, mask, cached=None, use_cache=False):
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)[:, -tokens.size(1):]
    stack.context(mask, use_cache)
    with mixed_precision():
        result = model.model(input_ids=tokens, attention_mask=mask, position_ids=positions,
                             past_key_values=cached, use_cache=use_cache)
        top = model.lm_head(result.last_hidden_state[:, -1:, :])[:, 0, :]
    return top.float(), result.past_key_values


def pad(rows, pad_id):
    width = max(map(len, rows))
    inputs = [([pad_id] * (width - len(row)) + row) for row in rows]
    mask = [([0] * (width - len(row)) + [1] * len(row)) for row in rows]
    return torch.tensor(inputs, device="cuda"), torch.tensor(mask, device="cuda", dtype=torch.bool)


def cache_check(model, stack, tokenizer):
    texts = ["Causal attention selects earlier tokens.", "One two three."]
    rows = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    token, mask = pad(rows, tokenizer.pad_token_id)
    stack.reset()
    full, _ = next_logits(model, stack, token, mask)
    _, cache = next_logits(model, stack, token[:, :-1], mask[:, :-1], use_cache=True)
    last, _ = next_logits(model, stack, token[:, -1:], mask, cache, use_cache=True)
    stack.reset()
    delta = (full - last).abs()
    assert torch.isfinite(last).all() and delta.mean().item() < .2, float(delta.mean())
    return {"mean_abs": float(delta.mean()), "max_abs": float(delta.max()),
            "same_argmax": int((full.argmax(-1) == last.argmax(-1)).sum()), "cases": len(rows)}


def generate(model, stack, tokenizer, rows, task, *, max_new_tokens=None, trace=None, numeric_stop_pattern=None,
             stop_on_content=True, stop_on_repetition=False):
    x, mask = pad(rows, tokenizer.pad_token_id)
    outputs = [[] for _ in rows]
    done = [False] * len(rows)
    limit = LIMIT[task] if max_new_tokens is None else max_new_tokens
    finished = ["budget"] * len(rows) if trace is not None else None
    number_stop = numeric_stop_pattern or re.compile(r"####\s*-?\d[\d,.]*\s*\n")
    stop = model.generation_config.eos_token_id
    stop = {stop} if isinstance(stop, int) else set(stop)
    stack.reset()
    logits, past = next_logits(model, stack, x, mask, use_cache=True)
    for step in range(limit):
        next_tokens = logits.argmax(-1)
        tokens = next_tokens.tolist()
        for j, token in enumerate(tokens):
            if done[j]:
                continue
            if token in stop:
                done[j] = True
                if finished is not None:
                    finished[j] = "eos"
            else:
                outputs[j].append(token)
        if stop_on_repetition and step + 1 == 384:
            from kam_math_repetition import repetitive

            for j, out in enumerate(outputs):
                if not done[j] and repetitive(tokenizer.decode(out, skip_special_tokens=True)):
                    done[j] = True
                    if finished is not None:
                        finished[j] = "repetition"
        if stop_on_content and (step + 1) % 16 == 0:
            for j, out in enumerate(outputs):
                if not done[j]:
                    decoded = tokenizer.decode(out, skip_special_tokens=True)
                    if task == "gsm8k":
                        done[j] = bool(number_stop.search(decoded))
                    else:
                        done[j] = decoded.count("```") >= 2
                    if done[j] and finished is not None:
                        finished[j] = "pattern"
        if all(done) or step + 1 == limit:
            break
        token = next_tokens.unsqueeze(1)
        mask = torch.cat((mask, torch.ones_like(token, dtype=torch.bool)), dim=1)
        logits, past = next_logits(model, stack, token, mask, past, use_cache=True)
    stack.reset()
    if trace is not None:
        trace.extend(zip(outputs, finished, strict=True))
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if task != "gsm8k":
        decoded = [m.group(1) if (m := CODE_BLOCK.search(s)) else s for s in decoded]
    return decoded


def evaluate(args):
    model, tokenizer = load_model(args.root)
    stack = AdapterStack(model, args.arm)
    fingerprint = (args.root / "model-sha256.txt").read_text()
    identity = {}
    adapter_sha = None
    if args.arm != "pretrained":
        adapter_path = args.root / f"adapter-{args.arm}.pt"
        checkpoint = torch.load(adapter_path, map_location="cuda", weights_only=True)
        identity = checkpoint["identity"]
        assert identity["arm"] == args.arm and identity["model_fingerprints"] == fingerprint
        assert identity["revision"] == REVISION
        stack.modules.load_state_dict(checkpoint["adapter"])
        adapter_sha = sha(adapter_path)
    stack.modules.eval()
    metadata = {
        "protocol": "qwen-pretrained-v1", "arm": args.arm, "model": MODEL_ID, "revision": REVISION,
        "model_fingerprints": fingerprint, "adapter_sha256": adapter_sha, "training": identity,
        "new_trainable_parameters": sum(p.numel() for p in stack.modules.parameters()),
    }
    score_task(args.root / "data", args.root / f"{args.task}-{args.arm}.json",
               args.task, args.batch, model, tokenizer, stack, metadata)


def score_task(data, output_path, task, batch_size, model, tokenizer, stack, metadata, *, numeric_audit=None):
    source, cases = load_cases(data, task)
    parity = cache_check(model, stack, tokenizer)
    prompts = [question(task, case, i) for i, case in enumerate(cases)]
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": raw}],
                                           tokenize=False, add_generation_prompt=True) for raw, _ in prompts]
    encodings = tokenizer(texts, add_special_tokens=False)["input_ids"]
    lengths = [len(ids) for ids in encodings]
    assert all(n <= CONTEXT for n in lengths), "unexpected prompt above the registered limit"
    order = sorted(range(len(cases)), key=lambda i: (lengths[i], i))
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    assert all(len(ids) == 1 for ids in letters)
    records = []
    for start in range(0, len(order), batch_size):
        indices = order[start:start+batch_size]
        rows = [encodings[i] for i in indices]
        if task in LIMIT:
            outputs = generate(model, stack, tokenizer, rows, task)
        else:
            x, mask = pad(rows, tokenizer.pad_token_id)
            logits, _ = next_logits(model, stack, x, mask)
            outputs = logits[:, [ids[0] for ids in letters]].argmax(-1).tolist()
        for i, output in zip(indices, outputs, strict=True):
            row = {"index": i, "prompt_tokens": lengths[i]}
            if task == "gsm8k":
                gold = numerical_answer(cases[i]["answer"])
                assert gold is not None
                answer = numerical_answer(output)
                row.update(correct=answer == gold, parsed=answer is not None)
                if numeric_audit is not None:
                    row.update(numeric_audit(output, gold, answer))
            elif task in ("humaneval", "mbpp"):
                row["completion"] = output
            else:
                row.update(correct=output == prompts[i][1], choice=output)
                if task == "mmlu":
                    row["subject"] = cases[i]["subject"]
            records.append(row)
        if start % (batch_size * 5) == 0:
            print(task, metadata["arm"], start + len(indices), len(cases), flush=True)
    metadata = {
        **metadata, "task": task, "source_sha256": sha(source), "cache_check": parity,
        "gpu": torch.cuda.get_device_name(), "context_tokens": CONTEXT,
        "total_backbone_parameters": sum(p.numel() for p in model.parameters()),
    }
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"metadata": metadata,
                                     "records": sorted(records, key=lambda row: row["index"])}))
    temporary.replace(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--task", choices=EXPECTED, required=True)
    parser.add_argument("--batch", type=int, default=48)
    evaluate(parser.parse_args())
