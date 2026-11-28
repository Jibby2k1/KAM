#!/usr/bin/env python3
"""Check whether polling for EOS every eight steps accelerates exact generation."""
import argparse
from functools import partial
from pathlib import Path

import torch

from eval_kam_pretrained import generate, next_logits, pad
from kam_math_kl_protocol import write_json
from probe_kam_decoder_token_reuse import measure_panels

POLL_INTERVAL = 8


def grouped_generate(model, stack, tokenizer, rows, task, *, max_new_tokens, trace, stop_on_content, surplus):
    assert task == "gsm8k" and not stop_on_content
    x, mask = pad(rows, tokenizer.pad_token_id)
    outputs = [[] for _ in rows]
    done = [False] * len(rows)
    reasons = ["budget"] * len(rows)
    eos = model.generation_config.eos_token_id
    stop = {eos} if isinstance(eos, int) else set(eos)
    stack.reset()
    logits, past = next_logits(model, stack, x, mask, use_cache=True)
    pending = []
    for step in range(1, max_new_tokens + 1):
        tokens = logits.argmax(-1)
        pending.append(tokens)
        if step % POLL_INTERVAL == 0 or step == max_new_tokens:
            for recorded in torch.stack(pending).tolist():
                for i, token in enumerate(recorded):
                    if done[i]:
                        continue
                    if token in stop:
                        done[i] = True
                        reasons[i] = "eos"
                    else:
                        outputs[i].append(token)
            pending.clear()
            if all(done) or step == max_new_tokens:
                break
        token = tokens.unsqueeze(1)
        mask = torch.cat((mask, torch.ones_like(token, dtype=torch.bool)), dim=1)
        logits, past = next_logits(model, stack, token, mask, past, use_cache=True)
    stack.reset()
    needed = max(len(output) + int(reason == "eos") for output, reason in zip(outputs, reasons, strict=True))
    assert 0 <= step - needed < POLL_INTERVAL
    surplus.append(step - needed)
    trace.extend(zip(outputs, reasons, strict=True))
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def probe(root, work, arm):
    surplus = []
    modes = {"per_token": generate, "grouped_eos": partial(grouped_generate, surplus=surplus)}
    name, provenance, panels = measure_panels(root, work, arm, modes)
    assert len(surplus) == 2 * (1 + 8)
    write_json(work / f"eos-poll-{name}.json", {**provenance, "panels": panels,
                "protocol": "kam-eos-poll-eight-v1", "poll_interval": POLL_INTERVAL,
                "surplus_batch_steps": {"64": surplus[:2], "1": surplus[2:]}})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("work", type=Path)
    cli.add_argument("arm", choices=("kam", "dot"))
    args = cli.parse_args()
    probe(args.root, args.work, args.arm)
