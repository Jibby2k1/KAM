#!/usr/bin/env python3
"""Intervene on the read budget of frozen live-token and fixed-bank Llama adapters."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from zipfile import ZipFile

from audit_kam_natural_data import rank, stable, words
from eval_kam_pretrained_endpoint_selection import literal
from kam_adapter_preservation import digest, key
from kam_math_kl_protocol import test_runs, write_json
from kam_musique_cached_retrieval import GPU, clip, display, visible_gold
from prepare_kam_musique_endpoint_pairs import candidate
from prepare_kam_native_musique import MEMBERS, canonical_paragraph
from prepare_kam_natural_router import title_key

PROTOCOL = "kam-frozen-routing-bottlenecks-v1"
MODES = {"pretrained": ("native",), "kam": ("native", "top16", "full_bank"),
         "dot": ("native", "explicit_dense", "top4", "top16")}


def prepare(source, recent, results, reservations, output, title_study):
    newest = json.loads(recent.read_text())["pairs"]
    paired = newest + [p for path in reservations for p in json.loads(path.read_text())]
    used_rows = [q for p in paired for q in p["questions"]]
    used_ids = {q["id"] for q in used_rows}
    titles_only = json.loads(title_study.read_text())
    used_ids.update(r["id"] for split in ("fit", "heldout") for r in titles_only[split]["records"])
    used_words = {tuple(words(q["question"])) for q in used_rows}
    used_titles = {title_key(d["title"]) for q in used_rows for d in q["evidence"]}
    used_families = {p["family"] for p in paired}
    outcomes = json.loads(gzip.decompress(results.read_bytes()))
    base = outcomes["routes-0-untouched-pretrained"]["records"]
    first = {r["question"] for r in base if r["dense"]["first_correct"]}
    assert all(r["dense"]["first_correct"] == (r["question"] in first) for r in base)
    rows = []
    for pair in newest:
        for i, q in enumerate(pair["questions"]):
            if q["id"] in first:
                rows.append({"cohort": "replay_" + pair["split"], "family": pair["family"], **q,
                             "foils": [pair["questions"][1-i]["evidence"][1]], "original_slot": i})
    with ZipFile(source) as archive, archive.open(MEMBERS["dev"]) as stream:
        dev = sorted((json.loads(line) for line in stream), key=lambda row: stable(PROTOCOL + row["id"]))
    new_families = set()
    for raw in dev:
        word_key = tuple(words(raw["question"]))
        if raw["id"] in used_ids or word_key in used_words:
            continue
        q = candidate(raw)
        if not q or q["hops"] != 2:
            continue
        family = stable(q["family"])
        titles = {title_key(d["title"]) for d in q["evidence"]}
        if family in used_families or family in new_families or titles & used_titles:
            continue
        foils = []
        for p in raw["paragraphs"]:
            title, body = p["title"], p["paragraph_text"]
            if p["is_supporting"] or title_key(title) in titles | used_titles:
                continue
            doc = {"id": stable("\x1f".join(canonical_paragraph(p))), "title": title, "body": body}
            if not any(literal(answer, display(doc)) for answer in q["answers"]):
                foils.append(doc)
        if not foils:
            continue
        query = q["question"] + " " + q["anchor"]
        ranked = rank(query, [display(d) for d in foils], limit=len(foils))
        anchored = [i for i in ranked if literal(q["anchor"], display(foils[i]))]
        foil = foils[(anchored or ranked)[0]]
        rows.append({**q, "cohort": "fresh_dev", "family": family, "foils": [foil], "original_slot": 0,
                     "foil_anchor": bool(anchored)})
        used_titles.update(titles | {title_key(foil["title"])})
        used_words.add(word_key)
        new_families.add(family)
        if len(new_families) == 96:
            break
    assert len(first) == sum(r["cohort"].startswith("replay_") for r in rows) and len(new_families) >= 32
    output.mkdir(parents=True, exist_ok=True)
    panel = output / "panel.json"
    write_json(panel, rows)
    write_json(output / "manifest.json", {"protocol": PROTOCOL, "source_sha256": digest(source),
               "previous_outcome_sha256": digest(results), "title_study_sha256": digest(title_study),
               "panel_sha256": digest(panel),
               "modes": MODES, "cases": [{"id": row["id"], "cohort": row["cohort"], "family": row["family"],
                                          "evidence_ids": [d["id"] for d in row["evidence"]],
                                          "foil_id": row["foils"][0]["id"], "foil_anchor": row.get("foil_anchor")}
                                         for row in rows]})


def evaluate(root, work, index):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import CONTEXT, next_logits, pad

    manifest, panel, selection = [work / f"{p}.json" for p in ("manifest", "panel", "selection")]
    config = json.loads(manifest.read_text())
    assert config["protocol"] == PROTOCOL and config["panel_sha256"] == digest(panel)
    recipe = json.loads(selection.read_text())["selected_recipe"]
    assert recipe == "lower_lr_kl05"
    run = test_runs(recipe)[index]
    modes = MODES[run[2]]
    model, tokenizer, stack, metadata = load_checkpoint(root, *run)
    assert torch.cuda.get_device_name() == GPU
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
    assert all(len(letter) == 1 for letter in letters) and letters[0] != letters[1]
    labels = torch.tensor([letter[0] for letter in letters], device="cuda")
    projection = model.lm_head.weight.detach().index_select(0, labels)

    def apply(mode, tokens, mask):
        for layer in stack.modules:
            if layer.arm == "dot":
                layer.diagnostic_attention = {"native": None, "explicit_dense": 0, "top4": 4, "top16": 16}[mode]
            else:
                layer.router.top_k = {"native": 4, "top16": 16, "full_bank": layer.keys.size(0)}[mode]
        start = perf_counter_ns()
        logits, _ = next_logits(model, stack, tokens, mask, projection=projection)
        a, b = logits[0].tolist()
        torch.cuda.synchronize()
        return b - a, (perf_counter_ns() - start) / 1e6

    records, excluded = [], []
    for q in json.loads(panel.read_text()):
        if not visible_gold(tokenizer, q):
            assert q["cohort"] == "fresh_dev"
            excluded.append(q["id"])
            continue
        first, gold = [clip(tokenizer, display(p)) for p in q["evidence"]]
        foil = clip(tokenizer, display(q["foils"][0]))
        for slot in (0, 1):
            a, b = (gold, foil) if slot == 0 else (foil, gold)
            message = ("Select the passage that answers the question. Use earlier evidence to identify the right entity "
                       "when it is available. Reply with A or B.\n\nQuestion: " + q["question"] +
                       "\n\nEarlier passage:\n" + first + "\n\nPassage A:\n" + a + "\n\nPassage B:\n" + b + "\n\nChoice:")
            text = tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                                 add_generation_prompt=True)
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert len(ids) <= CONTEXT
            tensors = pad([ids], tokenizer.pad_token_id)
            if not records:
                for mode in modes:
                    apply(mode, *tensors)
            observations = {mode: [] for mode in modes}
            for repeat in range(2):
                shift = (len(records) + repeat) % len(modes)
                for mode in modes[shift:] + modes[:shift]:
                    observations[mode].append(apply(mode, *tensors))
            item = {"id": q["id"], "expected": slot, "tokens": len(ids), "scores": {}}
            for mode, durations in observations.items():
                margin = durations[0][0]
                assert all(m == margin for m, _ in durations)
                item["scores"][mode] = {"margin": margin, "correct": margin != 0 and int(margin > 0) == slot,
                                        "median_ms": median(t for _, t in durations)}
            records.append(item)
    write_json(work / f"probe-{key(*run)}.json", {"metadata": {**metadata, "gpu": GPU, "protocol": PROTOCOL,
               "manifest_sha256": digest(manifest), "selection_sha256": digest(selection), "excluded": excluded},
               "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("source", "recent", "results", "output"):
        prep.add_argument(name, type=Path)
    prep.add_argument("reservations", nargs="+", type=Path)
    prep.add_argument("--title-study", required=True, type=Path)
    gpu = commands.add_parser("evaluate")
    gpu.add_argument("root", type=Path)
    gpu.add_argument("work", type=Path)
    gpu.add_argument("index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.recent, args.results, args.reservations, args.output, args.title_study)
    else:
        evaluate(args.root, args.work, args.index)
