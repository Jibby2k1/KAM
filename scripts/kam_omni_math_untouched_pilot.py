#!/usr/bin/env python3
"""Freeze and measure an untouched-backbone difficulty pilot on Omni-MATH."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from time import monotonic

from kam_adapter_preservation import digest
from kam_math_kl_protocol import normalized, write_json

PROTOCOL = "kam-omni-math-untouched-pilot-v1"
BANDS = ("at_most_2", "above_2_to_4", "above_4_to_6", "above_6")
PER_BAND = 12


def prepare(source, prior, prm_source, output):
    from audit_kam_omni_math_fixture import (
        SOURCE_SHA256, SCALAR, difficulty_band, gold_status, prior_prompts, read_jsonl,
    )
    from grade_kam_full_math500 import check_environment

    check_environment()
    assert digest(source) == SOURCE_SHA256
    rows = read_jsonl(source)
    keys = [normalized(row["problem"]) for row in rows]
    counts = Counter(keys)
    used = set().union(*prior_prompts(prior, prm_source).values())
    groups = defaultdict(list)
    for row, key in zip(rows, keys, strict=True):
        if counts[key] != 1 or key in used or not SCALAR.fullmatch(row["answer"].strip().strip("$")):
            continue
        if gold_status(row["answer"]) != "plain_scalar":
            continue
        ident = hashlib.sha256((PROTOCOL + "\n" + row["problem"]).encode()).hexdigest()
        band = difficulty_band(row["difficulty"])
        groups[band].append({"id": ident, "band": band, "difficulty": row["difficulty"],
                             "source": row["source"], "problem": row["problem"], "answer": row["answer"]})
    assert sum(map(len, groups.values())) == 1939
    fixture = [row for band in BANDS for row in sorted(groups[band], key=lambda row: row["id"])[:PER_BAND]]
    output.mkdir(parents=True, exist_ok=True)
    panel = output / "pilot.json"
    write_json(panel, fixture)
    write_json(output / "manifest.json", {
        "protocol": PROTOCOL, "source_sha256": SOURCE_SHA256, "fixture_sha256": digest(panel),
        "cases": [{key: val for key, val in row.items() if key not in ("answer", "problem")} for row in fixture],
    })


def load_panel(work):
    panel, manifest = work / "pilot.json", work / "manifest.json"
    registered = json.loads(manifest.read_text())
    assert registered["protocol"] == PROTOCOL and registered["fixture_sha256"] == digest(panel)
    rows = json.loads(panel.read_text())
    assert [row["id"] for row in registered["cases"]] == [row["id"] for row in rows]
    assert Counter(row["band"] for row in rows) == dict.fromkeys(BANDS, PER_BAND)
    return rows, digest(manifest)


def generate_untouched(root, work):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_full_math500 import BUDGET, PROMPT_SUFFIX
    from eval_kam_pretrained import CONTEXT, generate

    rows, manifest_sha256 = load_panel(work)
    model, tokenizer, stack, metadata = load_checkpoint(root, 0, "untouched", "pretrained")
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": row["problem"] + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True,
    ) for row in rows]
    encoded = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    assert max(map(len, encoded)) + BUDGET <= CONTEXT
    order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]), i))
    trace = []
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    torch.cuda.synchronize()
    start = monotonic()
    texts = generate(model, stack, tokenizer, [encoded[i] for i in order], "gsm8k",
                     max_new_tokens=BUDGET, trace=trace, stop_on_content=False)
    torch.cuda.synchronize()
    outcomes = [None] * len(rows)
    for i, completion, (tokens, stop) in zip(order, texts, trace, strict=True):
        outcomes[i] = {"id": rows[i]["id"], "prompt_tokens": len(encoded[i]), "output_tokens": len(tokens),
                       "stop": stop, "completion": completion}
    write_json(work / "untouched.json", {
        "metadata": {**metadata, "protocol": PROTOCOL, "manifest_sha256": manifest_sha256, "gpu": gpu,
                     "budget": BUDGET, "prompt_suffix": PROMPT_SUFFIX, "seconds": monotonic() - start},
        "records": outcomes,
    })


def grade(work, raw, output):
    from grade_kam_full_math500 import GRADER_VERSIONS, check_environment, grade_prediction, parse_gold

    check_environment()
    rows, manifest_sha256 = load_panel(work)
    outcomes = json.loads(raw.read_text())
    metadata = outcomes["metadata"]
    assert metadata["manifest_sha256"] == manifest_sha256 and metadata["protocol"] == PROTOCOL
    assert (metadata["seed"], metadata["recipe"], metadata["arm"]) == (0, "untouched", "pretrained")
    output_rows = []
    for row, scored in zip(rows, outcomes["records"], strict=True):
        assert row["id"] == scored["id"]
        assert len(gold := parse_gold(row["answer"])) == 1
        features = {**scored}
        completion = features.pop("completion")
        issues = []
        measured = grade_prediction(gold, completion, issues)
        output_rows.append({**features, **measured, "band": row["band"], "issues": issues})
    write_json(output, {"metadata": {**metadata, "grader_versions": GRADER_VERSIONS,
                                    "generation_sha256": digest(raw)}, "records": output_rows})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    for name in ("source", "prior", "prm_source", "output"):
        prep.add_argument(name, type=Path)
    gpu = modes.add_parser("generate")
    gpu.add_argument("root", type=Path)
    gpu.add_argument("work", type=Path)
    grading = modes.add_parser("grade")
    for name in ("work", "raw", "output"):
        grading.add_argument(name, type=Path)
    args = cli.parse_args()
    if args.mode == "prepare":
        prepare(args.source, args.prior, args.prm_source, args.output)
    elif args.mode == "generate":
        generate_untouched(args.root, args.work)
    else:
        grade(args.work, args.raw, args.output)
