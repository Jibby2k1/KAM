#!/usr/bin/env python3
"""Measure how benchmark behavior changes as a trained residual is scaled."""
import argparse
import json
import random
from pathlib import Path

import torch

from eval_kam_pretrained import LETTERS, load_cases, next_logits, pad, question
from kam_adapter_preservation import digest, key, layers_for, load_registered_model, model_identity
from kam_pretrained_adapters import AdapterStack
from train_kam_adapter_preservation import LENGTH, corpus, validation

SCALE = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED = 731221
TASKS = ("gpqa", "mmlu")
MMLU_ITEMS = 2048
CASE_SEED = 87613


def fixtures(root, tokenizer):
    prepared = {}
    for task in TASKS:
        source, rows = load_cases(root / "data", task)
        selected = list(range(len(rows)))
        if task == "mmlu":
            selected = sorted(random.Random(CASE_SEED).sample(selected, MMLU_ITEMS))
        prompts = [question(task, rows[i], i) for i in selected]
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                               add_generation_prompt=True) for prompt, _ in prompts]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        prepared[task] = {
            "source_sha256": digest(source), "ids": selected, "encoded": encoded,
            "gold": [gold for _, gold in prompts],
        }
    return prepared


@torch.inference_mode()
def predict(model, stack, tokenizer, panel):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    assert all(len(i) == 1 for i in ids)
    encoded = panel["encoded"]
    order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]), i))
    outputs = [None] * len(order)
    for start in range(0, len(order), 64):
        batch = order[start:start+64]
        x, mask = pad([encoded[i] for i in batch], tokenizer.pad_token_id)
        logits, _ = next_logits(model, stack, x, mask)
        choices = logits[:, [i[0] for i in ids]]
        predictions = choices.argmax(-1).tolist()
        for i, prediction in zip(batch, predictions, strict=True):
            outputs[i] = prediction
    return outputs


def measure(root, output, recipe, arm):
    model, tokenizer = load_registered_model(root)
    model_id, revision, _ = model_identity(root)
    stack = AdapterStack(model, arm, seed=SEED, layers=layers_for(model))
    adapter = root / f"adapter-{key(SEED, recipe, arm)}.pt"
    fitted = torch.load(adapter, map_location="cuda", weights_only=True)
    assert (fitted["identity"]["seed"], fitted["identity"]["recipe"], fitted["identity"]["arm"]) == (SEED, recipe, arm)
    assert fitted["identity"]["model"] == model_id and fitted["identity"]["revision"] == revision
    stack.modules.load_state_dict(fitted["adapter"])
    stack.modules.eval()
    trained_up = [m.up.weight.detach().clone() for m in stack.modules]
    panels = fixtures(root, tokenizer)
    manifest = json.loads((root / "token-manifest.json").read_text())
    data = corpus(root, "validation", manifest)
    offsets = torch.arange(LENGTH + 1, device="cuda")
    stack.active = False
    reference = {task: predict(model, stack, tokenizer, panel) for task, panel in panels.items()}
    stack.active = True
    observations = []
    for strength in SCALE:
        with torch.no_grad():
            for module, up in zip(stack.modules, trained_up, strict=True):
                module.up.weight.copy_(up * strength)
        stack.reset()
        stats = validation(model, stack, data, offsets)
        predictions = {task: predict(model, stack, tokenizer, panel) for task, panel in panels.items()}
        if strength == 0:
            assert predictions == reference
        observations.append({"scale": strength, "validation": stats, "choices": predictions})
        print(model_id, recipe, arm, strength, stats["forward_kl"], flush=True)
    for panel in panels.values():
        del panel["encoded"]
    payload = {
        "model": model_id, "revision": revision, "seed": SEED, "recipe": recipe, "arm": arm,
        "adapter_sha256": digest(adapter), "gpu": torch.cuda.get_device_name(),
        "panels": panels, "untouched": reference, "observations": observations,
    }
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, separators=(",", ":")))
    temp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recipe", choices=("reference", "high_lr_kl"), required=True)
    parser.add_argument("--arm", choices=("kam", "dot"), required=True)
    a = parser.parse_args()
    measure(a.root, a.output, a.recipe, a.arm)
