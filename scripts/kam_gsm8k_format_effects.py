#!/usr/bin/env python3
"""Freeze and run a three-prompt GSM8K format intervention on unused questions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import monotonic

from kam_adapter_preservation import digest, key, model_identity
from kam_math_kl_protocol import BATCH_SIZE, BUDGET, MARKERS, SOURCES, normalized, patterns, score, test_runs, write_json

PROTOCOL = "kam-gsm8k-format-effects-v1"
PROMPT_SUFFIX = "\n\nSolve the problem. Explain your reasoning and state your final answer clearly."
PROMPTS = {"neutral": PROMPT_SUFFIX}
for label, marker in MARKERS.items():
    PROMPTS[label] = PROMPT_SUFFIX + f" Write your final answer on its own line beginning with '{marker} '."


def prepare(source, prior_manifest, output):
    for name, sha in SOURCES.items():
        assert digest(source / name) == sha, name
    assert digest(prior_manifest) == "e49fd394a1c17150808f42d66999dd5c7588fa2b04bba26e6133c524c77f5f0c"
    locked = json.loads(prior_manifest.read_text())
    used_indices = {int(ident.split(":")[1]) for ident in locked["fixtures"]["dev"]["ids"]}
    train = [json.loads(s) for s in (source / "gsm8k-train.jsonl").read_text().splitlines()]
    old_test = [json.loads(s) for s in (source / "gsm8k-test.jsonl").read_text().splitlines()]
    math = [json.loads(s) for s in (source / "math500-test.jsonl").read_text().splitlines()]
    svamp = json.loads((source / "SVAMP.json").read_text())
    excluded = {normalized(row["question"]) for row in old_test}
    excluded.update(normalized(row["problem"]) for row in math)
    excluded.update(normalized(row["Body"] + " " + row["Question"]) for row in svamp)
    normalized_train = [normalized(row["question"]) for row in train]
    counts = Counter(normalized_train)
    eligible = [i for i, words in enumerate(normalized_train)
                if i not in used_indices and counts[words] == 1 and words not in excluded]
    ranked = sorted(eligible, key=lambda i: hashlib.sha256(f"{PROTOCOL}:test:{i}".encode()).digest())
    rows = []
    for index in sorted(ranked[:512]):
        item = train[index]
        gold = int(item["answer"].rpartition("#### ")[-1].strip().replace(",", ""))
        rows.append({"id": f"train:{index}", "problem": item["question"], "answer": gold})
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "test.json"
    assert len(rows) == 512
    write_json(fixture, rows)
    write_json(output / "manifest.json", {"protocol": PROTOCOL, "fixture_sha256": digest(fixture),
                                          "prior_manifest_sha256": digest(prior_manifest),
                                          "sources": SOURCES, "prompts": PROMPTS, "budget": BUDGET,
                                          "batch_size": BATCH_SIZE, "ids": [row["id"] for row in rows]})


def load_fixture(work):
    manifest = json.loads((work / "manifest.json").read_text())
    fixture = work / "test.json"
    assert manifest["protocol"] == PROTOCOL and digest(fixture) == manifest["fixture_sha256"]
    assert manifest["budget"] == BUDGET and manifest["prompts"] == PROMPTS
    rows = json.loads(fixture.read_text())
    assert [row["id"] for row in rows] == manifest["ids"]
    return manifest, rows


def evaluate(root, work, index):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import CONTEXT, generate

    selection_path = work / "selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["model"] == model_identity(root)[0]
    run = test_runs(selection["selected_recipe"])[index]
    ident = key(*run)
    model, tokenizer, stack, metadata = load_checkpoint(root, *run)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    manifest, dataset = load_fixture(work)
    encoded = {}
    for mode, suffix in PROMPTS.items():
        texts = [
            tokenizer.apply_chat_template([{"role": "user", "content": row["problem"] + suffix}],
                                          tokenize=False, add_generation_prompt=True)
            for row in dataset
        ]
        encoded[mode] = tokenizer(texts, add_special_tokens=False)["input_ids"]
    assert max(len(ids) for family in encoded.values() for ids in family) + BUDGET <= CONTEXT
    order = sorted(range(len(dataset)), key=lambda i: (len(encoded["neutral"][i]), i))
    records = [{"id": row["id"]} for row in dataset]
    timings = {}
    for mode, family in encoded.items():
        start_time = monotonic()
        for start in range(0, len(order), BATCH_SIZE):
            group = order[start:start+BATCH_SIZE]
            trace = []
            texts = generate(model, stack, tokenizer, [family[i] for i in group], "gsm8k",
                             max_new_tokens=BUDGET, trace=trace, stop_on_content=False)
            for i, text, (tokens, stop) in zip(group, texts, trace, strict=True):
                records[i][mode] = {"completion": text, "prompt_tokens": len(family[i]),
                                    "output_tokens": len(tokens), "stop": stop}
            print(ident, mode, start + len(group), len(order), flush=True)
        torch.cuda.synchronize()
        timings[mode] = monotonic() - start_time
    write_json(work / f"format-{ident}.json", {
        "metadata": {**metadata, "gpu": gpu, "protocol": PROTOCOL, "fixture_sha256": manifest["fixture_sha256"],
                     "manifest_sha256": digest(work / "manifest.json"), "selection_sha256": digest(selection_path),
                     "seconds": timings},
        "records": records,
    })


def grade(work, generation, output):
    from grade_kam_full_math500 import GRADER_PROTOCOL, GRADER_VERSIONS, check_environment, grade_prediction, parse_gold

    check_environment()
    manifest, data = load_fixture(work)
    payload = json.loads(generation.read_text())
    assert payload["metadata"]["fixture_sha256"] == manifest["fixture_sha256"]
    assert [r["id"] for r in payload["records"]] == manifest["ids"]
    strict = {mode: patterns(marker)[0] for mode, marker in MARKERS.items()}
    rows = []
    diagnostics = []
    for item, original in zip(data, payload["records"], strict=True):
        gold = parse_gold(item["answer"])
        assert len(gold) == 1
        row = {"id": item["id"]}
        for mode in PROMPTS:
            fields = original[mode]
            completion = fields["completion"]
            result = {field: fields[field] for field in ("prompt_tokens", "output_tokens", "stop")}
            issues = []
            result["symbolic"] = grade_prediction(gold, completion, issues)
            diagnostics.extend({"id": item["id"], "mode": mode, "error": error} for error in issues)
            if mode in strict:
                parsed, correct = score(completion, item["answer"], strict[mode])
                result["strict"] = {"parsed": parsed, "correct": correct}
                result["marker_present"] = MARKERS[mode].casefold() in completion.casefold()
            row[mode] = result
        rows.append(row)
    write_json(output, {"metadata": {**payload["metadata"], "generation_sha256": digest(generation),
                                     "grader_protocol": GRADER_PROTOCOL, "grader_versions": GRADER_VERSIONS,
                                     "grader_diagnostics": diagnostics},
                        "records": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cli = commands.add_parser("prepare")
    prepare_cli.add_argument("source", type=Path)
    prepare_cli.add_argument("prior_manifest", type=Path)
    prepare_cli.add_argument("output", type=Path)
    evaluate_cli = commands.add_parser("evaluate")
    evaluate_cli.add_argument("root", type=Path)
    evaluate_cli.add_argument("work", type=Path)
    evaluate_cli.add_argument("index", type=int)
    grade_cli = commands.add_parser("grade")
    grade_cli.add_argument("work", type=Path)
    grade_cli.add_argument("generation", type=Path)
    grade_cli.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.prior_manifest, args.output)
    elif args.command == "evaluate":
        evaluate(args.root, args.work, args.index)
    else:
        grade(args.work, args.generation, args.output)
