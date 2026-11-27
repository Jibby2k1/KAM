#!/usr/bin/env python3
"""Rescore unchanged GSM8K completions with a conservative terminal-number fallback."""
import argparse
import json
import re
from decimal import Decimal
from pathlib import Path

from kam_adapter_preservation import FACTORIAL_RUNS, key

NUMBER = r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
BOXED = re.compile(r"\\boxed\s*\{\s*" + NUMBER + r"\s*\}\s*(?:[.$*]|`|\\\]|\s)*\Z")
CUE = re.compile(r"(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?:is\s*:?(?:\s+|\Z)|:\s*)",
                 flags=re.IGNORECASE)
TERMINAL = re.compile(r"[\s$*`]*" + NUMBER + r"[\s$*`]*[.!]?[\s$*`]*\Z")


def score_terminal_number(completion, gold, strict):
    if strict is not None:
        return {"terminal_correct": strict == gold, "terminal_parser": "marked"}
    text = completion.rstrip()
    boxed = BOXED.search(text)
    if boxed is not None:
        return {"terminal_correct": Decimal(boxed[1].replace(",", "")) == Decimal(gold),
                "terminal_parser": "boxed"}
    line = text.split("\n")[-1].strip("*` ").removesuffix(".")
    cue = CUE.match(line)
    candidate = TERMINAL.fullmatch(line[cue.end():] if cue else line)
    if candidate is not None:
        return {"terminal_correct": Decimal(candidate[1].replace(",", "")) == Decimal(gold),
                "terminal_parser": "answer_cue" if cue else "terminal_line"}
    return {"terminal_correct": False, "terminal_parser": "unrecognized"}


def audit(root, index):
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import score_task

    seed, recipe, arm = FACTORIAL_RUNS[index]
    name = key(seed, recipe, arm)
    strict = json.loads((root / f"gsm8k-{name}.json").read_text())
    model, tokenizer, stack, metadata = load_checkpoint(root, seed, recipe, arm)
    out = root / f"format-gsm8k-{name}.json"
    score_task(root / "data", out, "gsm8k", 64, model, tokenizer, stack, metadata,
               numeric_audit=score_terminal_number)
    replay = json.loads(out.read_text())
    for field in ("source_sha256", "adapter_sha256", "model_fingerprints", "context_tokens", "gpu"):
        assert strict["metadata"][field] == replay["metadata"][field], field
    mismatches = 0
    for recorded, raw in zip(strict["records"], replay["records"], strict=True):
        assert (raw["index"], raw["prompt_tokens"]) == (recorded["index"], recorded["prompt_tokens"])
        assert raw["correct"] <= raw["terminal_correct"]
        mismatches += any(raw[field] != recorded[field] for field in ("correct", "parsed"))
    print(name, "strict item flag mismatches", mismatches, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    audit(args.root, args.index)
