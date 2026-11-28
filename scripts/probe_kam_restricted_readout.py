#!/usr/bin/env python3
"""Time two-letter readout with the full vocabulary head and a sliced head."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from eval_kam_pretrained_endpoint_selection import prompts
from kam_adapter_preservation import digest, key
from kam_math_kl_protocol import test_runs, write_json

PROTOCOL = "kam-llama-fixed-choice-readout-v1"


def evaluate(root, work, index):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import next_logits, pad

    manifest_path, panel, selection = (work / name for name in ("manifest.json", "panel.json", "selection.json"))
    manifest = json.loads(manifest_path.read_text())
    recipe = json.loads(selection.read_text())["selected_recipe"]
    assert manifest["protocol"] == PROTOCOL and manifest["panel_sha256"] == digest(panel)
    assert recipe == "lower_lr_kl05"
    run = test_runs(recipe)[index]
    model, tokenizer, stack, meta = load_checkpoint(root, *run)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert isinstance(model.lm_head, torch.nn.Linear) and model.lm_head.bias is None
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
    assert all(len(ids) == 1 for ids in letters) and letters[0] != letters[1]
    labels = torch.tensor([ids[0] for ids in letters], device="cuda")
    projection = model.lm_head.weight.detach().index_select(0, labels)

    def run_projection(x, mask, restricted):
        logits, _ = next_logits(model, stack, x, mask, projection=projection if restricted else None)
        if not restricted:
            logits = logits.index_select(-1, labels)
        return logits.tolist()

    def measure(request, reverse):
        trials = {False: [], True: []}
        schedule = (reverse, not reverse, not reverse, reverse)
        x, mask = pad(request, tokenizer.pad_token_id)
        for restricted in schedule:
            torch.cuda.synchronize()
            start = perf_counter_ns()
            values = run_projection(x, mask, restricted)
            torch.cuda.synchronize()
            trials[restricted].append((values, (perf_counter_ns() - start) / 1e6))
        result = {}
        for mode, restricted in (("full", False), ("sliced", True)):
            a, b = trials[restricted]
            assert a[0] == b[0]
            result[mode] = {"logits": a[0], "median_ms": median((a[1], b[1]))}
        return result

    pairs = json.loads(panel.read_text())
    accepted = set(manifest["pairs"])
    pairs = [p for p in pairs if p["id"] in accepted]
    assert {p["id"] for p in pairs} == accepted
    rows, encoded = prompts(tokenizer, pairs, both_earlier=True)
    first = encoded["true_first"]
    for j in (0, len(rows) // 2, len(rows) - 1):
        measure([first[j]], bool(j % 2))
        measure([first[j], first[j ^ 1]], bool(j % 2))
    results = []
    for j, row in enumerate(rows):
        for slot, mode in enumerate(("true_first", "true_second")):
            view = encoded[mode]
            results.append({**row, "earlier_slot": slot,
                            "width_1": measure([view[j]], bool((j + slot) % 2)),
                            "width_2": measure([view[j], view[j ^ 1]], bool((j + slot) % 2))})
    write_json(work / f"readout-{key(*run)}.json", {"metadata": {**meta, "gpu": gpu, "protocol": PROTOCOL,
               "manifest_sha256": digest(manifest_path), "selection_sha256": digest(selection)}, "records": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate",))
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    evaluate(args.root, args.work, args.index)
