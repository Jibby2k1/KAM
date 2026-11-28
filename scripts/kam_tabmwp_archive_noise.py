#!/usr/bin/env python3
"""Evaluate tabular arithmetic with a matched, explicitly archived distractor table."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from time import monotonic

from kam_adapter_preservation import LLAMA_ID, digest, key
from kam_math_kl_protocol import BATCH_SIZE, normalized, test_runs, write_json

PROTOCOL = "kam-tabmwp-archive-noise-v1"
UNRELATED_PROTOCOL = "kam-tabmwp-unrelated-length-control-v1"
REVISION = "869bae8ffb8288ccc3102293b3090ec2aa17f618"
SOURCE_HASHES = {
    "test": "c47607dad36477725087cb4e5317fe1e89a69dfed404eca430673bda36fda45c",
    "dev": "e20fa61be57f66fb574c7fe97e8ef958d8589bfe10371947d7214ac3bf51e904",
}
MODES = ("current_only", "archive_first", "archive_last")
GRADES = (5, 6, 7, 8)
PER_GRADE = 32
BUDGET = 384
SLOT = "KAM_TABULAR_RECORDS_HERE"


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def table_identity(row):
    return normalized(row["table_title"] or ""), normalized(row["table"])


def relation(row):
    header = row["table"].splitlines()[0]
    return normalized(row["table_title"] or ""), normalized(header)


def eligible_table(row):
    return row["table_title"] and 3 <= row["row_num"] <= 16 and len(row["table"]) <= 2000


def prepare(source, prior, prm_source, output):
    from audit_kam_omni_math_fixture import prior_prompts
    from grade_kam_full_math500 import VERIFY, check_environment, parse_gold, verify

    check_environment()
    test, dev = (source / f"problems_{split}.json" for split in ("test", "dev"))
    assert (digest(test), digest(dev)) == (SOURCE_HASHES["test"], SOURCE_HASHES["dev"])
    test_rows, dev_rows = json.loads(test.read_text()), json.loads(dev.read_text())
    assert len(test_rows) == 7686 and set(test_rows).isdisjoint(dev_rows)
    known = set().union(*prior_prompts(prior, prm_source).values())
    donors, dev_tables = defaultdict(list), {table_identity(row) for row in dev_rows.values()}
    for ident, row in dev_rows.items():
        if eligible_table(row):
            donors[(row["grade"], row["row_num"], relation(row))].append((ident, row))
    counters = Counter()
    groups = defaultdict(list)
    for ident, row in test_rows.items():
        answer = row["answer"].strip()
        if row["grade"] not in GRADES or not eligible_table(row) or row["ques_type"] != "free_text":
            continue
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", answer) is None:
            continue
        if table_identity(row) in dev_tables or normalized(row["question"]) in known:
            counters["exact_existing_overlap"] += 1
            continue
        title_header = relation(row)
        matching = donors[(row["grade"], row["row_num"], title_header)]
        if not matching:
            counters["no_shape_and_title_match"] += 1
            continue
        parsed = parse_gold(answer)
        if len(parsed) != 1 or not verify(parsed, parsed, **VERIFY):
            counters["unverified_gold"] += 1
            continue
        family = hash_text(repr(title_header))
        groups[row["grade"]].append((ident, row, matching, family))

    panel, used_tables, used_donors = [], set(), set()
    for grade in GRADES:
        for ident, row, choices, family in sorted(groups[grade], key=lambda vals: hash_text(f"{PROTOCOL}:{vals[0]}")):
            table = table_identity(row)
            if table in used_tables:
                continue
            unused = [(i, r) for i, r in choices if i not in used_donors and table_identity(r) not in used_tables]
            if not unused:
                continue
            donor, archived = min(unused, key=lambda vals: hash_text(f"{PROTOCOL}:{ident}:{vals[0]}"))
            used_tables.update((table, table_identity(archived)))
            used_donors.add(donor)
            panel.append({"id": ident, "donor": donor, "grade": grade, "family": family,
                          "answer": row["answer"].strip(), "question": row["question"],
                          "title": row["table_title"], "current": row["table"], "archived": archived["table"]})
            if sum(item["grade"] == grade for item in panel) == PER_GRADE:
                break
    assert len(panel) == len({item["id"] for item in panel}) == PER_GRADE * len(GRADES)
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "panel.json"
    write_json(fixture, panel)
    write_json(output / "manifest.json", {
        "protocol": PROTOCOL, "revision": REVISION, "source_hashes": SOURCE_HASHES, "panel_sha256": digest(fixture),
        "modes": MODES, "budget": BUDGET, "batch_size": BATCH_SIZE,
        "eligible_by_grade": {grade: len(groups[grade]) for grade in GRADES}, "audit_counts": counters,
        "cases": [{k: r[k] for k in ("id", "donor", "grade", "family")} for r in panel],
    })


def load_panel(work, unrelated=False):
    manifest = work / ("unrelated-manifest.json" if unrelated else "manifest.json")
    registered = json.loads(manifest.read_text())
    source = work / ("unrelated-panel.json" if unrelated else "panel.json")
    protocol = UNRELATED_PROTOCOL if unrelated else PROTOCOL
    assert registered["protocol"] == protocol and registered["panel_sha256"] == digest(source)
    rows = json.loads(source.read_text())
    fields = ("id", "donor", "grade", "family")
    if unrelated:
        fields += ("control_donor",)
        assert registered["n"] == len(rows)
    else:
        assert (tuple(registered["modes"]), registered["budget"], registered["batch_size"]) == (MODES, BUDGET, BATCH_SIZE)
    assert [{k: r[k] for k in fields} for r in rows] == registered["cases"]
    return rows, digest(manifest)


def encode_case(tokenizer, row):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    prompt = ("Use the current table to answer the question. An archived table, if present, refers to a separate "
              "record.\n\n" + SLOT + "\nQuestion: " + row["question"] +
              "\n\nSolve the problem. Explain your reasoning and state your final answer clearly.")
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)
    before, after = map(encode, chat.split(SLOT))
    current = encode("Current table: " + row["title"] + "\n" + row["current"] + "\n\n")
    archive_title = row.get("archived_title", row["title"])
    archive = encode("Archived table: " + archive_title + "\n" + row["archived"] + "\n\n")
    return {"current_only": [*before, *current, *after],
            "archive_first": [*before, *archive, *current, *after],
            "archive_last": [*before, *current, *archive, *after]}


def evaluate(root, work, index, unrelated=False):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import CONTEXT, generate

    data, manifest = load_panel(work, unrelated)
    modes = ("archive_last",) if unrelated else MODES
    controls = {}
    if unrelated:
        controls = {row["id"]: row for row in data}
        original, original_manifest = load_panel(work / "baseline")
        expected_original = json.loads((work / "unrelated-manifest.json").read_text())["original_manifest_sha256"]
        assert original_manifest == expected_original
        data = [controls.get(row["id"], row) for row in original]
    selection_path = work / "selection.json"
    selection = json.loads(selection_path.read_text())
    assert (selection["model"], selection["selected_recipe"], selection["selected_kl_weight"]) == (
        LLAMA_ID, "lower_lr_kl05", .5,
    )
    run = test_runs(selection["selected_recipe"])[index]
    model, tokenizer, stack, metadata = load_checkpoint(root, *run)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    encoded = [encode_case(tokenizer, row) for row in data]
    assert all(len(e["archive_first"]) == len(e["archive_last"]) <= CONTEXT - BUDGET for e in encoded)
    order = sorted(range(len(data)), key=lambda i: (len(encoded[i]["current_only"]), i))
    records = [{k: row[k] for k in ("id", "grade", "family")} for row in data]
    seconds = {}
    for mode in modes:
        torch.cuda.synchronize()
        start = monotonic()
        for offset in range(0, len(order), BATCH_SIZE):
            indices = order[offset:offset+BATCH_SIZE]
            inputs = [encoded[i][mode] for i in indices]
            trace = []
            messages = generate(model, stack, tokenizer, inputs, "gsm8k", max_new_tokens=BUDGET,
                                trace=trace, stop_on_content=False)
            for i, prompt, message, (tokens, stop) in zip(indices, inputs, messages, trace, strict=True):
                records[i][mode] = {"prompt_tokens": len(prompt), "output_tokens": len(tokens),
                                    "stop": stop, "completion": message}
        torch.cuda.synchronize()
        seconds[mode] = monotonic() - start
        print(key(*run), mode, seconds[mode], flush=True)
    parity = {}
    if unrelated:
        original_path = work / "baseline" / f"tabmwp-{key(*run)}.json"
        baseline = json.loads(original_path.read_text())
        assert baseline["metadata"]["manifest_sha256"] == original_manifest
        assert [r["id"] for r in records] == [r["id"] for r in baseline["records"]]
        for new, old in zip(records, baseline["records"], strict=True):
            assert new["archive_last"]["prompt_tokens"] == old["archive_last"]["prompt_tokens"]
            if new["id"] not in controls:
                assert new["archive_last"] == old["archive_last"], new["id"]
        parity = {"unmatched_unchanged": len(records) - len(controls),
                  "baseline_generation_sha256": digest(original_path)}
        records = [row for row in records if row["id"] in controls]
    write_json(work / f"tabmwp-{key(*run)}.json", {
        "metadata": {**metadata, **parity, "protocol": UNRELATED_PROTOCOL if unrelated else PROTOCOL,
                     "manifest_sha256": manifest, "gpu": gpu,
                     "selection_sha256": digest(selection_path), "seconds": seconds}, "records": records,
    })


def grade(work, raw_path, output, terminal=False, unrelated=False):
    from grade_kam_full_math500 import GRADER_VERSIONS, check_environment, grade_prediction, parse_gold, terminal_lines

    check_environment()
    data, manifest = load_panel(work, unrelated)
    protocol = UNRELATED_PROTOCOL if unrelated else PROTOCOL
    modes = ("archive_last",) if unrelated else MODES
    raw = json.loads(raw_path.read_text())
    assert raw["metadata"]["protocol"] == protocol and raw["metadata"]["manifest_sha256"] == manifest
    for row, record in zip(data, raw["records"], strict=True):
        assert row["id"] == record["id"]
        assert len(gold := parse_gold(row["answer"])) == 1
        for mode in modes:
            response = record[mode]
            issues = []
            text = response.pop("completion")
            response.update(grade_prediction(gold, terminal_lines(text) if terminal else text, issues))
            response["issues"] = issues
    raw["metadata"].update(grader_versions=GRADER_VERSIONS, generation_sha256=digest(raw_path))
    if terminal:
        raw["metadata"]["scoring_window"] = "terminal_three_lines"
    write_json(output, raw)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    operations = cli.add_subparsers(dest="operation", required=True)
    prep = operations.add_parser("prepare")
    for name in ("source", "prior", "prm_source", "output"):
        prep.add_argument(name, type=Path)
    gpu = operations.add_parser("evaluate")
    gpu.add_argument("root", type=Path)
    gpu.add_argument("work", type=Path)
    gpu.add_argument("index", type=int, choices=range(7))
    gpu.add_argument("--unrelated", action="store_true")
    grading = operations.add_parser("grade")
    for name in ("work", "raw", "output"):
        grading.add_argument(name, type=Path)
    grading.add_argument("--terminal", action="store_true")
    grading.add_argument("--unrelated", action="store_true")
    args = cli.parse_args()
    if args.operation == "prepare":
        prepare(args.source, args.prior, args.prm_source, args.output)
    elif args.operation == "evaluate":
        evaluate(args.root, args.work, args.index, args.unrelated)
    else:
        grade(args.work, args.raw, args.output, args.terminal, args.unrelated)
