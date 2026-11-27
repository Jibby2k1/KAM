#!/usr/bin/env python3
"""Measure teacher-to-student KL at the assistant cue on fixed benchmark inputs."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from eval_kam_pretrained import LETTERS, next_logits, pad
from kam_adapter_preservation import FACTORIAL_TRAIN_RUNS, digest, key, layers_for, load_registered_model, model_identity
from kam_pretrained_adapters import AdapterStack
from probe_kam_adapter_strength import fixtures


def matrix(model, stack, tokenizer, pane):
    encoded = pane["encoded"]
    order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]), i))
    result = torch.empty((len(order), model.config.vocab_size), dtype=torch.float32, device="cuda")
    for start in range(0, len(order), 64):
        members = order[start:start+64]
        ids, mask = pad([encoded[i] for i in members], tokenizer.pad_token_id)
        logits, _ = next_logits(model, stack, ids, mask)
        result[members] = logits
    return result


def aggregate_rows(full, choices, teacher, flags, gold):
    n = len(gold)
    stats = []
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        student_p = full[start:stop].log_softmax(-1)
        target = teacher[start:stop]
        kl = F.kl_div(student_p, target, reduction="none", log_target=True).sum(-1).tolist()
        student_choices = choices[start:stop].log_softmax(-1).tolist()
        pred = choices[start:stop].argmax(-1).tolist()
        # The teacher's four-choice log distribution is compared separately
        # because full-vocabulary probability can move outside A through D.
        for off, prediction in enumerate(pred):
            index = start + off
            stats.append({"full_kl": kl[off], "choice_log_p": student_choices[off],
                          "choice": prediction, "correct": prediction == gold[index],
                          "teacher_correct": flags[index]})
    return stats


@torch.inference_mode()
def probe(root, task, output):
    model, tokenizer = load_registered_model(root)
    model_id, revision, _ = model_identity(root)
    pane = fixtures(root, tokenizer)[task]
    idx = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]
    base = AdapterStack(model, "pretrained")
    teacher = matrix(model, base, tokenizer, pane)
    teacher_choices = teacher[:, idx].log_softmax(-1)
    expected = teacher_choices.argmax(-1).tolist()
    flags = [choice == g for choice, g in zip(expected, pane["gold"], strict=True)]
    for start in range(0, len(expected), 128):
        teacher[start:start+128] = teacher[start:start+128].log_softmax(-1)
    results = {}
    for seed, recipe, arm in FACTORIAL_TRAIN_RUNS:
        name = key(seed, recipe, arm)
        file = root / f"adapter-{name}.pt"
        stack = AdapterStack(model, arm, seed=seed, layers=layers_for(model))
        try:
            checkpoint = torch.load(file, map_location="cuda", weights_only=True)
            md = checkpoint["identity"]
            assert (md["seed"], md["recipe"], md["arm"], md["model"], md["revision"]) == (seed, recipe, arm, model_id, revision)
            stack.modules.load_state_dict(checkpoint["adapter"])
            stack.modules.eval()
            logits = matrix(model, stack, tokenizer, pane)
            stats = aggregate_rows(logits, logits[:, idx], teacher, flags, pane["gold"])
            results[name] = {"adapter_sha256": digest(file), "rows": stats}
            del logits
            print(model_id, task, name, sum(r["correct"] for r in stats), flush=True)
        finally:
            stack.reset()
            stack.close()
    del pane["encoded"]
    payload = {"task": task, "model": model_id, "revision": revision, "gpu": torch.cuda.get_device_name(),
               "panel": pane, "untouched_choices": expected, "teacher_choice_log_p": teacher_choices.tolist(),
               "runs": results}
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, separators=(",", ":")))
    temp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("task", choices=("gpqa", "mmlu"))
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    probe(a.root, a.task, a.output)
