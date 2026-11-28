#!/usr/bin/env python3
"""Evaluate the locked adapter checkpoints on independent PG-19 and HotpotQA inputs."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from eval_kam_pretrained import next_logits, pad
from kam_adapter_preservation import digest, load_registered_model, model_identity
from kam_live_token_factorial import CHECKPOINTS, GPU, LiveTokenStack, RUNS
from kam_math_kl_protocol import write_json
from kam_pretrained_adapters import mixed_precision
from prepare_kam_live_transfer import PROTOCOL, SCORE_TOKENS


@torch.inference_mode()
def measure(model, tokenizer, stack, data):
    books = data["pg19"]
    frames = torch.tensor([row for book in books for row in book["windows"]], device="cuda")
    assert frames.size(1) == SCORE_TOKENS + 1
    nll = []
    stack.context(None, False)
    for windows in frames.split(4):
        with mixed_precision():
            hidden = model.model(input_ids=windows[:, :-1], use_cache=False).last_hidden_state
            logits = model.lm_head(hidden)
        target = windows[:, 1:]
        nll.extend(F.cross_entropy(logits.flatten(0, 1).float(), target.reshape(-1),
                                   reduction="none").reshape_as(target).mean(-1).tolist())
    book_scores = []
    offset = 0
    for book in books:
        count = len(book["windows"])
        book_scores.append({"id": book["id"], "nll": nll[offset:offset+count]})
        offset += count
    assert offset == len(nll)
    letters = [tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
    assert all(len(x) == 1 for x in letters) and letters[0] != letters[1] and model.lm_head.bias is None
    projection = model.lm_head.weight.index_select(0, torch.tensor([x[0] for x in letters], device="cuda"))
    support = []
    for row in data["retrieval"]:
        inputs, mask = pad([row["ids"]], tokenizer.pad_token_id)
        margins, _ = next_logits(model, stack, inputs, mask, projection=projection)
        a, b = margins[0].tolist()
        margin = b-a
        support.append({"id": row["id"], "expected": row["expected"], "cohort": row["cohort"], "margin": margin,
                        "correct": margin != 0 and int(margin > 0) == row["expected"]})
    stack.reset()
    return {"pg19": book_scores, "retrieval": support}


def evaluate(root, work, index):
    manifest, inputs = work / "input-manifest.json", work / "inputs.json"
    locked = json.loads(manifest.read_text())
    model_id, revision, _ = model_identity(root)
    assert locked["protocol"] == PROTOCOL and locked["inputs_sha256"] == digest(inputs)
    assert (model_id, revision) == (locked["model"], locked["revision"])
    assert torch.cuda.get_device_name() == GPU
    data = json.loads(inputs.read_text())
    seed, arm = RUNS[index] if index < len(RUNS) else (0, "pretrained")
    name = f"{seed}-{arm}"
    model, tokenizer = load_registered_model(root)
    stack = LiveTokenStack(model, seed, arm)
    result = {"input_manifest_sha256": digest(manifest), "model": model_id, "revision": revision, "gpu": GPU,
              "arm": arm, "seed": seed, "checkpoints": {}}
    epochs = (0,) if arm == "pretrained" else CHECKPOINTS
    training = json.loads((work / f"train-{name}.json").read_text()) if arm != "pretrained" else None
    for step in epochs:
        if step:
            path = work / f"adapter-{name}-{step}.pt"
            receipt, = [row for row in training["history"] if row["step"] == step]
            checksum = digest(path)
            assert receipt["adapter_sha256"] == checksum
            ckpt = torch.load(path, map_location="cuda", weights_only=True)
            identity = ckpt["identity"]
            assert (identity["seed"], identity["arm"], identity["model"], identity["revision"], ckpt["step"]) == (
                seed, arm, model_id, revision, step,
            )
            assert identity["protocol_sha256"] == digest(work / "training.json")
            stack.modules.load_state_dict(ckpt["adapter"])
        else:
            checksum = None
        result["checkpoints"][str(step)] = {"adapter_sha256": checksum, **measure(model, tokenizer, stack, data)}
        print(name, step, flush=True)
    write_json(work / f"eval-{name}.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("index", type=int, choices=range(len(RUNS)+1))
    args = p.parse_args()
    evaluate(args.root, args.work, args.index)
