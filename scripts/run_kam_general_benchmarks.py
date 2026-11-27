#!/usr/bin/env python3
"""Score the registered TinyStories checkpoint pair on five text test sets."""
import argparse
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from kam.transformer.baselines import build_baseline

PAIR = "tinystories_v2_128mib:71001"
ARMS = ("T-KAM-F", "T-WIDE")
EXPECTED = {"humaneval": 164, "mbpp": 500, "gpqa": 198, "mmlu": 14042, "gsm8k": 1319}
GEN_LIMIT = {"humaneval": 384, "mbpp": 384, "gsm8k": 256}
LETTERS = "ABCD"
NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
FINAL_NUMBER = re.compile(r"####\s*(" + NUMBER + r")")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(root, arm):
    with (root / "manifest.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    pair = {row["architecture"]: row for row in rows if row.get("pair_id") == PAIR}
    for key in ("corpus_id", "data_seed", "minimum_tokens", "sequence_length",
                "batch_size", "training_seed", "precision", "target_parameter_budget"):
        assert pair[ARMS[0]][key] == pair[ARMS[1]][key], key
    row = pair[arm]
    path = root / "checkpoints" / f"{row['row_id']}_seed{row['training_seed']}_final.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["row"] == row
    assert checkpoint["metrics"]["tokens"] == 50_001_920
    model, _ = build_baseline(
        arm, scale=row["scale"], vocab_size=256, max_seq_len=row["sequence_length"],
        num_supports=row["num_supports"], top_k=row["top_k"],
        seed=row["training_seed"], target_parameters=row["target_parameter_budget"],
    )
    model.load_state_dict(checkpoint["model"])
    info = {
        "arm": arm, "row_id": row["row_id"], "pair": PAIR, "sha256": digest(path),
        "tokens": checkpoint["metrics"]["tokens"], "seed": row["training_seed"],
        "context_bytes": row["sequence_length"], "parameters": sum(p.numel() for p in model.parameters()),
        "validation_loss": checkpoint["metrics"]["validation_loss"], "data_seed": row["data_seed"],
    }
    return model.to("cuda").eval(), info


def load_cases(data, task):
    source = data / f"{task}.parquet"
    if task == "gpqa":
        source = data / "gpqa.csv"
        with source.open(newline="", encoding="utf-8-sig") as stream:
            cases = list(csv.DictReader(stream))
    else:
        cases = pq.read_table(source).to_pylist()
    assert len(cases) == EXPECTED[task], (task, len(cases))
    return source, cases


def make_prompt(task, case, index):
    if task == "humaneval":
        return case["prompt"], None
    if task == "mbpp":
        return f"# Write a Python function.\n# {case['text']}\n", None
    if task == "gsm8k":
        return f"Question: {case['question']}\nAnswer (end with #### and the number):", None
    if task == "gpqa":
        options = [case["Correct Answer"]] + [case[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
        order = list(range(4))
        random.Random(71001 + index).shuffle(order)
        gold = order.index(0)
        options = [options[i] for i in order]
        question = case["Question"]
    else:
        options, gold, question = case["choices"], case["answer"], case["question"]
    choices = "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, options, strict=True))
    return f"{question}\n{choices}\nAnswer: ", gold


@torch.inference_mode()
def last_logits(model, prompts, context):
    # Right padding follows the last real byte. Causality keeps it outside the readout.
    tails = [p[-context:] for p in prompts]
    lens = [len(p) for p in tails]
    width = max(lens)
    x = torch.tensor([list(p) + [0] * (width - len(p)) for p in tails], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    positions = torch.tensor(lens, device=x.device) - 1
    return logits[torch.arange(len(tails), device=x.device), positions].float()


def generate(model, prompts, context, limit):
    completions = [bytearray() for _ in prompts]
    active = list(range(len(prompts)))
    for _ in range(limit):
        if not active:
            break
        histories = [prompts[i][-context:] + completions[i] for i in active]
        tokens = last_logits(model, histories, context).argmax(dim=-1).tolist()
        following = []
        for i, token in zip(active, tokens, strict=True):
            completions[i].append(token)
            if token and not completions[i].endswith(b"\n\n\n"):
                following.append(i)
        active = following
    return [bytes(c).split(b"\x00", 1)[0].decode("utf-8", errors="replace") for c in completions]


def disable_unused_diagnostics(model, prompts, context):
    # The pinned checkpoint source computes route statistics even with diagnostics
    # disabled. Forward only uses the route's selected indices and weights.
    from kam.memory import sparse_kam
    original = sparse_kam.routing_diagnostics
    samples = [prompts[:1], prompts[:7], prompts[:32]]
    started = time.perf_counter()
    reference = [last_logits(model, batch, context) for batch in samples]
    original_seconds = time.perf_counter() - started
    reference_prefix = generate(model, samples[-1], context, 8)

    def no_diagnostics(*_args, **_kwargs):
        return {}

    sparse_kam.routing_diagnostics = no_diagnostics
    try:
        started = time.perf_counter()
        optimized = [last_logits(model, batch, context) for batch in samples]
        for expected, actual in zip(reference, optimized, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        optimized_seconds = time.perf_counter() - started
        assert generate(model, samples[-1], context, 8) == reference_prefix
    except BaseException:
        sparse_kam.routing_diagnostics = original
        raise
    print(f"Unused route statistics omitted after exact logits and 8-byte prefix checks; "
          f"three forwards {original_seconds:.3f}s -> {optimized_seconds:.3f}s", flush=True)


def numerical_answer(text):
    matches = FINAL_NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else None


def run(args):
    source, cases = load_cases(args.data, args.task)
    model, metadata = load_model(args.checkpoints, args.arm)
    context = metadata["context_bytes"]
    records = []
    prompts = [make_prompt(args.task, row, i) for i, row in enumerate(cases)]
    lengths = [len(prompt.encode("utf-8")) for prompt, _ in prompts]
    if args.omit_route_diagnostics:
        disable_unused_diagnostics(model, [prompt.encode("utf-8") for prompt, _ in prompts], context)
    for batch_index, start in enumerate(range(0, len(cases), args.batch)):
        if batch_index % args.shards != args.shard:
            continue
        stop = min(start + args.batch, len(cases))
        encoded = [prompts[i][0].encode("utf-8") for i in range(start, stop)]
        if args.task in GEN_LIMIT:
            outputs = generate(model, encoded, context, GEN_LIMIT[args.task])
        else:
            scores = last_logits(model, encoded, context)[:, list(LETTERS.encode("ascii"))]
            outputs = scores.argmax(dim=-1).tolist()
        for index, prediction in zip(range(start, stop), outputs, strict=True):
            record = {"index": index, "prompt_bytes": lengths[index]}
            if args.task == "gsm8k":
                gold = numerical_answer(cases[index]["answer"])
                assert gold is not None
                answer = numerical_answer(prediction)
                record.update(correct=answer == gold, parsed=answer is not None)
            elif args.task in ("humaneval", "mbpp"):
                record["completion"] = prediction
            else:
                record["correct"] = prediction == prompts[index][1]
                record["choice"] = prediction
                if args.task == "mmlu":
                    record["subject"] = cases[index]["subject"]
            records.append(record)
        if batch_index % (args.shards * 25) == args.shard:
            print(f"{args.task} {args.arm}: {stop}/{len(cases)}", flush=True)
    metadata.update(task=args.task, source_sha256=digest(source),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name())
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"metadata": metadata, "records": records}))
    temporary.replace(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=EXPECTED, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--omit-route-diagnostics", action="store_true")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if args.batch < 1 or not 0 <= args.shard < args.shards:
        parser.error("batch must be positive and shard must be in [0, shards)")
    run(args)
