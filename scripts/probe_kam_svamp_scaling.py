#!/usr/bin/env python3
"""Evaluate original and scaled quantities in unambiguous two-number SVAMP items."""
import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

AMOUNT = re.compile(r"\b\d+\b")
EXCLUDED = re.compile(r"\b(?:years?|months?|weeks?|days?|hours?|minutes?|seconds?|old|percent|dozens?|"
                      r"half|double|triple|first|second|third|each|per)\b|\d[.,]\d|[$€%]|[a-z]\d|\d[a-z]", re.I)


def variants(data):
    result = []
    for index, row in enumerate(data):
        original = row["Body"].strip() + " " + row["Question"].strip()
        if EXCLUDED.search(original):
            continue
        counts = [int(amount) for amount in AMOUNT.findall(original)]
        if len(counts) != 2 or min(counts) < 2 or max(counts) > 300 or row["Answer"] <= 0:
            continue
        expr = ast.parse(row["Equation"].strip(), mode="eval").body
        if not isinstance(expr, ast.BinOp) or not isinstance(expr.op, (ast.Add, ast.Sub)):
            continue
        operands = (expr.left, expr.right)
        if any(not isinstance(node, ast.Constant) or type(node.value) not in (int, float)
               or node.value != int(node.value) for node in operands):
            continue
        terms = [int(node.value) for node in operands]
        if Counter(terms) != Counter(counts):
            continue
        sign = 1 if isinstance(expr.op, ast.Add) else -1
        assert terms[0] + sign * terms[1] == row["Answer"]
        multiplier = 2 + (hashlib.sha256(row["ID"].encode()).digest()[0] & 1)
        scaled = AMOUNT.sub(lambda match: str(multiplier * int(match[0])), original)
        label = multiplier * terms[0] + sign * multiplier * terms[1]
        assert label != row["Answer"]
        result.append({"index": index, "id": row["ID"], "type": row["Type"], "multiplier": multiplier,
                       "original": original, "scaled": scaled, "original_answer": int(row["Answer"]),
                       "scaled_answer": label})
    return result


def probe(root, index):
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import generate, numerical_answer, question
    from eval_kam_svamp_transfer import BATCH_SIZE, SVAMP_REVISION, SVAMP_SHA256
    from kam_adapter_preservation import FACTORIAL_RUNS, LLAMA_ID, digest, key

    source = root / "SVAMP.json"
    assert digest(source) == SVAMP_SHA256
    original = json.loads(source.read_text())
    rows = variants(original)
    assert len(rows) == 149
    seed, recipe, arm = FACTORIAL_RUNS[index]
    assert recipe in ("lower_lr", "untouched")
    model, tokenizer, stack, metadata = load_checkpoint(root, seed, recipe, arm)
    assert metadata["model"] == LLAMA_ID
    packed = {}
    for mode in ("original", "scaled"):
        prompts = [question("gsm8k", {"question": row[mode]}, row["index"])[0] for row in rows]
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                               add_generation_prompt=True) for prompt in prompts]
        packed[mode] = tokenizer(texts, add_special_tokens=False)["input_ids"]
    order = sorted(range(len(rows)), key=lambda i: (len(packed["original"][i]), i))
    outcomes = {row["index"]: {field: row[field] for field in ("index", "id", "type", "multiplier")}
                for row in rows}
    for start in range(0, len(order), BATCH_SIZE):
        group = order[start:start+BATCH_SIZE]
        for mode in ("original", "scaled"):
            generated = generate(model, stack, tokenizer, [packed[mode][i] for i in group], "gsm8k")
            for i, output in zip(group, generated, strict=True):
                row = rows[i]
                parsed = numerical_answer(output)
                answer = Decimal(parsed) if parsed is not None else None
                result = outcomes[row["index"]]
                result[f"{mode}_parsed"] = answer is not None
                result[f"{mode}_correct"] = answer == row[f"{mode}_answer"]
                if mode == "scaled":
                    result["scaled_selected_original_answer"] = answer == row["original_answer"]
    target = root / f"svamp-scaling-{key(seed, recipe, arm)}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps({"metadata": {**metadata, "source_revision": SVAMP_REVISION,
                                                   "source_sha256": SVAMP_SHA256},
                                      "records": [outcomes[i] for i in sorted(outcomes)]},
                                     separators=(",", ":")))
    temporary.replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    probe(args.root, args.index)
