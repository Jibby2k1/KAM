#!/usr/bin/env python3
"""Time a fixed repetition stop on disjoint prompts and verify unaffected traces."""
import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

from kam_adapter_preservation import LLAMA_ID, digest, key
from kam_math_answer_type_replication import load_replication
from kam_math_kl_protocol import write_json
from kam_math_repetition import repetitive

PROTOCOL = "kam-math-repeat-timing-v1"
SHAPES = (("kam", 1, 8), ("dot", 1, 8), ("kam", 8, 24), ("dot", 8, 24))


def profile(root, work, index):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_full_math500 import BUDGET, PREFIX_BUDGET, PROMPT_SUFFIX
    from eval_kam_pretrained import generate

    arm, batch, n_pairs = SHAPES[index]
    source = work / "math-replication.json"
    manifest = work / "manifest.json"
    data = load_replication(source, manifest)
    selection = work / "selection.json"
    selected = json.loads(selection.read_text())
    assert (selected["model"], selected["selected_recipe"], selected["selected_kl_weight"]) == (
        LLAMA_ID, "lower_lr_kl05", .5,
    )
    pair_order = sorted(range(len(data) // 2), key=lambda i: hashlib.sha256(
        f"{PROTOCOL}:{data[i * 2]['unique_id']}".encode()).digest())
    chosen = [data[i * 2 + off] for i in pair_order[:n_pairs] for off in range(2)]
    run = (731221, selected["selected_recipe"], arm)
    model, tokenizer, stack, metadata = load_checkpoint(root, *run)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": row["problem"] + PROMPT_SUFFIX}],
                                      tokenize=False, add_generation_prompt=True)
        for row in chosen
    ]
    encoded = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]), i))
    expected = {"original": None, "repeat": None}
    times = {label: [] for label in expected}
    for label in ("original", "repeat", "repeat", "original"):
        outputs = {}
        torch.cuda.synchronize()
        start_time = monotonic()
        for start in range(0, len(order), batch):
            group = order[start:start+batch]
            trace = []
            generate(model, stack, tokenizer, [encoded[i] for i in group], "gsm8k",
                     max_new_tokens=BUDGET, trace=trace, stop_on_content=False,
                     stop_on_repetition=label == "repeat")
            for i, (tokens, reason) in zip(group, trace, strict=True):
                outputs[i] = (tokens, reason)
        torch.cuda.synchronize()
        times[label].append(monotonic() - start_time)
        if expected[label] is not None:
            assert outputs == expected[label], label
        expected[label] = outputs
        print(arm, batch, label, times[label][-1], flush=True)
    checks = []
    for i, row in enumerate(chosen):
        full, full_stop = expected["original"][i]
        reduced, reduced_stop = expected["repeat"][i]
        test = len(full) >= PREFIX_BUDGET and repetitive(
            tokenizer.decode(full[:PREFIX_BUDGET], skip_special_tokens=True)
        )
        if test:
            assert reduced_stop == "repetition" and reduced == full[:PREFIX_BUDGET]
        else:
            assert (reduced, reduced_stop) == (full, full_stop)
        checks.append({"id": row["unique_id"], "prompt_tokens": len(encoded[i]),
                       "original_tokens": len(full), "original_stop": full_stop,
                       "repeat_tokens": len(reduced), "repeat_stop": reduced_stop})
    name = key(*run)
    write_json(work / f"timing-{batch}-{name}.json", {
        "metadata": {**metadata, "gpu": gpu, "protocol": PROTOCOL, "batch": batch,
                     "manifest_sha256": digest(manifest), "selection_sha256": digest(selection),
                     "timings_seconds": times},
        "records": checks,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("work", type=Path)
    cli.add_argument("index", type=int, choices=range(len(SHAPES)))
    args = cli.parse_args()
    profile(args.root, args.work, args.index)
