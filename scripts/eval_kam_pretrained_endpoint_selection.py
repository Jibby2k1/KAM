#!/usr/bin/env python3
"""Probe whether matched pretrained adapters use early MuSiQue evidence."""
import argparse
import gzip
import json
from pathlib import Path
import re
from time import monotonic

from kam_adapter_preservation import digest, key, model_identity
from kam_math_kl_protocol import test_runs, write_json

PROTOCOL = "kam-pretrained-endpoint-choice-v1"
MODES = ("endpoints", "true_prefix", "cross_prefix")
BATCH = 16
PLACEHOLDER = "KAM_PRIOR_PASSAGE_PLACEHOLDER"


def literal(term, text):
    return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE))


def prepare(source, original, output):
    registered = json.loads((original / "manifest.json").read_text())
    assert digest(source) == registered["fixture_sha256"]
    fixture = json.loads(gzip.decompress(source.read_bytes()))
    accepted = set()
    for shard in range(4):
        shard_path = original / f"endpoints-53016711-{shard}.json.gz"
        current = json.loads(gzip.decompress(shard_path.read_bytes()))
        assert current["fixture_sha256"] == registered["fixture_sha256"]
        accepted |= {item["pair"] for item in current["rows"] if item["hops"] == 2}
    pairs = [pair for pair in fixture["pairs"] if pair["id"] in accepted]
    assert len(pairs) == len({r["id"] for r in pairs}) == 84
    assert all(r["hops"] == 2 for r in pairs)
    assert len({r["family"] for r in pairs}) == 35
    output.mkdir(parents=True, exist_ok=True)
    file = output / "panel.json"
    write_json(file, pairs)
    manifest = {
        "protocol": PROTOCOL, "source_sha256": registered["source_sha256"],
        "paired_fixture_sha256": digest(source), "panel_sha256": digest(file),
        "batch_size": BATCH, "modes": MODES, "cases": [
            {"pair": pair["id"], "family": pair["family"], "question_ids": [q["id"] for q in pair["questions"]]}
            for pair in pairs
        ],
    }
    write_json(output / "manifest.json", manifest)


def prompts(tokenizer, pairs):
    def tokenize(text, limit=None):
        return tokenizer.encode(text, add_special_tokens=False, **(
            {"truncation": True, "max_length": limit} if limit is not None else {}
        ))

    def visible(doc, limit):
        return tokenize(doc["title"] + ". " + doc["body"], limit)

    def chat(message):
        return tokenizer.apply_chat_template([{"role": "user", "content": message}],
                                             tokenize=False, add_generation_prompt=True)

    def instruction(question, ends, earlier=None):
        task = ("Select the passage that answers the question. "
                "Use earlier evidence to identify the right entity when it is available. "
                "Reply with A or B.\n\nQuestion: " + question)
        if earlier is not None:
            task += "\n\nEarlier passage:\n" + earlier
        return task + "\n\nPassage A:\n" + ends[0] + "\n\nPassage B:\n" + ends[1] + "\n\nChoice:"

    all_cases, encoded = [], {mode: [] for mode in MODES}
    for pair in pairs:
        questions = pair["questions"]
        assert all(len(q["evidence"]) == 2 for q in questions)
        early = [visible(q["evidence"][0], 512) for q in questions]
        width = min(map(len, early))
        early = [ids[:width] for ids in early]
        last = [tokenizer.decode(visible(q["evidence"][1], 512), skip_special_tokens=True) for q in questions]
        for i, q in enumerate(questions):
            assert literal(q["anchor"], tokenizer.decode(early[i], skip_special_tokens=True)), q["id"]
            assert literal(q["anchor"], last[i]) and any(literal(a, last[i]) for a in q["answers"]), q["id"]
            for reverse in (False, True):
                ends = last[::-1] if reverse else last
                record = {"pair": pair["id"], "family": pair["family"], "question": q["id"],
                          "reverse": reverse, "expected": i ^ int(reverse)}
                all_cases.append(record)
                encoded["endpoints"].append(tokenize(chat(instruction(q["question"], ends))))
                head, tail = chat(instruction(q["question"], ends, PLACEHOLDER)).split(PLACEHOLDER)
                start, end = tokenize(head), tokenize(tail)
                encoded["true_prefix"].append([*start, *early[i], *end])
                encoded["cross_prefix"].append([*start, *early[1 - i], *end])
    return all_cases, encoded


def evaluate(root, work, index):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import CONTEXT, next_logits, pad

    source = work / "panel.json"
    manifest_path = work / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"] == PROTOCOL and manifest["panel_sha256"] == digest(source)
    selection_path = work / "selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["model"] == model_identity(root)[0]
    run = test_runs(selection["selected_recipe"])[index]
    name = key(*run)
    model, tokenizer, stack, metadata = load_checkpoint(root, *run)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    rows, encoded = prompts(tokenizer, json.loads(source.read_text()))
    assert len(rows) == 336 and max(len(v) for family in encoded.values() for v in family) <= CONTEXT
    assert list(map(len, encoded["true_prefix"])) == list(map(len, encoded["cross_prefix"]))
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
    assert all(len(v) == 1 for v in letters) and letters[0] != letters[1]
    ids = [v[0] for v in letters]
    order = sorted(range(len(rows)), key=lambda i: (len(encoded["true_prefix"][i]), i))
    timings = {}
    for mode, family in encoded.items():
        torch.cuda.synchronize()
        started = monotonic()
        for start in range(0, len(order), BATCH):
            idx = order[start:start+BATCH]
            x, mask = pad([family[i] for i in idx], tokenizer.pad_token_id)
            logits, _ = next_logits(model, stack, x, mask)
            choices = logits[:, ids].float()
            scores = (choices[:, 1] - choices[:, 0]).tolist()
            predicted = choices.argmax(-1).tolist()
            for i, raw, choice in zip(idx, scores, predicted, strict=True):
                rows[i][mode] = {"tokens": len(family[i]), "picked": choice, "b_minus_a": raw}
        torch.cuda.synchronize()
        timings[mode] = monotonic() - started
        print(name, mode, timings[mode], flush=True)
    write_json(work / f"endpoint-{name}.json", {
        "metadata": {**metadata, "gpu": gpu, "protocol": PROTOCOL, "manifest_sha256": digest(manifest_path),
                     "selection_sha256": digest(selection_path), "seconds": timings},
        "records": rows,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="operation", required=True)
    prep = command.add_parser("prepare")
    prep.add_argument("fixture", type=Path)
    prep.add_argument("original", type=Path)
    prep.add_argument("output", type=Path)
    evaluate_cli = command.add_parser("evaluate")
    evaluate_cli.add_argument("root", type=Path)
    evaluate_cli.add_argument("work", type=Path)
    evaluate_cli.add_argument("index", type=int)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.fixture, args.original, args.output)
    else:
        evaluate(args.root, args.work, args.index)
