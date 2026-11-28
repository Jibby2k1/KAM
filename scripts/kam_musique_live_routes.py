#!/usr/bin/env python3
"""Freeze new relation families and measure actual single-request GPU routes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from zipfile import ZipFile

from audit_kam_natural_data import rank, stable, words
from eval_kam_pretrained_endpoint_selection import earlier_choice_prompts, prompts
from kam_adapter_preservation import digest, key
from kam_math_kl_protocol import test_runs, write_json
from prepare_kam_musique_endpoint_pairs import candidate, compatible
from prepare_kam_native_musique import MEMBERS
from prepare_kam_natural_router import title_key

PROTOCOL = "kam-musique-live-routes-v1"
ROUTES = ("direct", "sequential", "mirrored")
LEXICAL_PROTOCOL = "kam-musique-lexical-route-v1"
LEXICAL_ROUTES = ("direct", "lexical")


def prepare(source, prior, output):
    old = json.loads(prior.read_text())
    reserved_family = {r["family"] for r in old}
    reserved = [q for pair in old for q in pair["questions"]]
    ids = {q["id"] for q in reserved}
    wordings = {tuple(words(q["question"])) for q in reserved}
    used_titles = {title_key(p["title"]) for q in reserved for p in q["evidence"]}
    buckets = defaultdict(list)
    with ZipFile(source) as archive:
        for split in ("dev", "train"):
            with archive.open(MEMBERS[split]) as data:
                for line in data:
                    row = json.loads(line)
                    if row["id"] in ids or tuple(words(row["question"])) in wordings:
                        continue
                    item = candidate(row)
                    if item and item["hops"] == 2 and stable(item["family"]) not in reserved_family:
                        buckets[split, item["family"]].append(item)
    pairs = []
    split_counts = defaultdict(int)
    for (split, family), options in sorted(buckets.items(), key=lambda kv: (kv[0][0], stable(PROTOCOL + kv[0][1]))):
        if split_counts[split] >= 96:
            continue
        ordered = sorted(options, key=lambda r: stable(PROTOCOL + r["id"]))
        accepted = 0
        for i, left in enumerate(ordered):
            if accepted >= 3 or split_counts[split] >= 96:
                break
            if left["id"] in ids:
                continue
            left_words = tuple(words(left["question"]))
            left_titles = {title_key(p["title"]) for p in left["evidence"]}
            if left_words in wordings or left_titles & used_titles:
                continue
            for right in ordered[i + 1:]:
                right_words = tuple(words(right["question"]))
                right_titles = {title_key(p["title"]) for p in right["evidence"]}
                if right["id"] in ids or right_words in wordings or right_words == left_words or right_titles & used_titles:
                    continue
                if compatible(left, right):
                    pairs.append({"id": stable(PROTOCOL + left["id"] + "|" + right["id"]),
                                  "family": stable(family), "split": split, "hops": 2, "questions": [left, right]})
                    used_titles.update(left_titles | right_titles)
                    wordings.update((left_words, right_words))
                    ids.update((left["id"], right["id"]))
                    accepted += 1
                    split_counts[split] += 1
                    break
    assert pairs and not reserved_family.intersection(p["family"] for p in pairs)
    output.mkdir(parents=True, exist_ok=True)
    panel = output / "panel.json"
    write_json(panel, pairs)
    write_json(output / "manifest.json", {
        "protocol": PROTOCOL, "source_sha256": digest(source), "members": MEMBERS,
        "prior_panel_sha256": digest(prior), "panel_sha256": digest(panel), "routes": ROUTES,
        "cases": [{"pair": p["id"], "family": p["family"], "split": p["split"],
                   "questions": [q["id"] for q in p["questions"]]}
                  for p in pairs],
    })


def evaluate(root, work, index, lexical=False):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import CONTEXT, next_logits, pad

    selection = work / "selection.json"
    chosen = json.loads(selection.read_text())
    assert chosen["selected_recipe"] == "lower_lr_kl05" and chosen["selected_kl_weight"] == .5
    run = test_runs(chosen["selected_recipe"])[index]
    frozen = work / "manifest.json"
    manifest = json.loads(frozen.read_text())
    panel = work / "panel.json"
    protocol, routes = (LEXICAL_PROTOCOL, LEXICAL_ROUTES) if lexical else (PROTOCOL, ROUTES)
    assert manifest["protocol"] == protocol and tuple(manifest["routes"]) == routes
    assert manifest["panel_sha256"] == digest(panel)
    pairs = json.loads(panel.read_text())
    model, tokenizer, stack, metadata = load_checkpoint(root, *run)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    # Condition eligibility on tokenizer visibility, never on model predictions.
    eligible, excluded = [], []
    for pair in pairs:
        try:
            prompts(tokenizer, [pair])
        except AssertionError as error:
            if str(error) not in {q["id"] for q in pair["questions"]}:
                raise
            excluded.append(pair["id"])
        else:
            eligible.append(pair)
    rows, direct = prompts(tokenizer, eligible, both_earlier=True)
    compare_rows, endpoint = prompts(tokenizer, eligible)
    early_rows, early = earlier_choice_prompts(tokenizer, eligible)
    assert rows == compare_rows == early_rows and len(rows) >= 4
    early = early["earlier_choice"]
    assert all(len(x) <= CONTEXT for x in early)
    assert all(len(x) <= CONTEXT for mode in (direct, endpoint) for batch in mode.values() for x in batch)
    letter = [tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
    assert all(len(x) == 1 for x in letter) and letter[0] != letter[1]
    labels = [x[0] for x in letter]
    lexical_inputs = {}
    if lexical:
        for pair in eligible:
            snippets = [q["evidence"][0]["title"] + ". " + q["evidence"][0]["body"] for q in pair["questions"]]
            ids = [tokenizer.encode(s, add_special_tokens=False)[:512] for s in snippets]
            width = min(map(len, ids))
            texts = [tokenizer.decode(row[:width], skip_special_tokens=True) for row in ids]
            for i, q in enumerate(pair["questions"]):
                lexical_inputs[q["id"]] = (q["question"], texts, i)

    def margins(requests):
        x, mask = pad(requests, tokenizer.pad_token_id)
        logits, _ = next_logits(model, stack, x, mask)
        choices = logits[:, labels].float()
        return (choices[:, 1] - choices[:, 0]).tolist()

    def route(j, slot, name):
        inputs = direct["true_first" if slot == 0 else "true_second"]
        if name == "direct":
            score, = margins([inputs[j]])
            return score, len(inputs[j]), None
        if name == "mirrored":
            score, swapped = margins([inputs[j], inputs[j ^ 1]])
            return score - swapped, len(inputs[j]) + len(inputs[j ^ 1]), None
        if name == "lexical":
            question, texts, original = lexical_inputs[rows[j]["question"]]
            candidates = texts if original == slot else texts[::-1]
            first, = rank(question, candidates, limit=1)
            mode = "true_prefix" if first == slot else "cross_prefix"
            score, = margins([endpoint[mode][j]])
            return score, len(endpoint[mode][j]), first == slot
        first_index = j if rows[j]["expected"] == slot else j ^ 1
        first_score, = margins([early[first_index]])
        mode = "true_prefix" if int(first_score > 0) == slot else "cross_prefix"
        score, = margins([endpoint[mode][j]])
        return score, len(early[first_index]) + len(endpoint[mode][j]), None

    warm = (0, len(rows) // 2, len(rows) - 1)
    for j in warm:
        for name in routes:
            route(j, 0, name)
    records = []
    for j, row in enumerate(rows):
        for slot in (0, 1):
            result = {**row, "earlier_slot": slot}
            order = (j * 2 + slot) % len(routes)
            for name in routes[order:] + routes[:order]:
                duration = []
                observations = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    start = perf_counter_ns()
                    observations.append(route(j, slot, name))
                    torch.cuda.synchronize()
                    duration.append((perf_counter_ns() - start) / 1e6)
                assert all(value == observations[0] for value in observations)
                margin, width, first = observations[0]
                result[name] = {"correct": margin != 0 and int(margin > 0) == row["expected"],
                                "margin": margin, "tokens": width, "median_ms": median(duration),
                                **({"first_correct": first} if first is not None else {})}
            records.append(result)
    write_json(work / f"routes-{key(*run)}.json", {"metadata": {**metadata, "gpu": gpu,
               "protocol": protocol, "selection_sha256": digest(selection), "manifest_sha256": digest(frozen),
               "excluded_pairs": excluded}, "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("source", "prior", "output"):
        prep.add_argument(name, type=Path)
    gpu = commands.add_parser("evaluate")
    gpu.add_argument("root", type=Path)
    gpu.add_argument("work", type=Path)
    gpu.add_argument("index", type=int)
    gpu.add_argument("--lexical", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.prior, args.output)
    else:
        evaluate(args.root, args.work, args.index, args.lexical)
