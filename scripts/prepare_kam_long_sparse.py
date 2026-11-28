#!/usr/bin/env python3
"""Prepare disjoint long PG-19 windows and position-controlled MuSiQue prompts."""
import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

from audit_kam_natural_data import rank, stable, words
from eval_kam_pretrained_endpoint_selection import literal
from kam_adapter_preservation import digest, model_identity
from kam_math_kl_protocol import write_json
from prepare_kam_musique_endpoint_pairs import candidate
from prepare_kam_native_musique import MEMBERS, canonical_paragraph
from prepare_kam_natural_router import title_key

PROTOCOL = "kam-long-sparse-v1"
CONTEXT = 4096
TARGETS = 1024
PG_WINDOWS = 3
CASES = 320
MARKER = "KAMDISTRACTORBLOCK"


def display(doc):
    return doc["title"] + ". " + doc["body"]


def freeze_retrieval(source, previous, work, *, count=CASES, excluded_titles=(), excluded_families=()):
    reserved = set(previous.read_text().split())
    with ZipFile(source) as archive:
        source_rows = {}
        for split in ("dev", "train"):
            with archive.open(MEMBERS[split]) as data:
                source_rows[split] = sorted(map(json.loads, data), key=lambda r: stable(PROTOCOL + r["id"]))
    bank = {}
    for row in source_rows["dev"]:
        for p in row["paragraphs"]:
            if not p["is_supporting"]:
                ident = stable("\x1f".join(canonical_paragraph(p)))
                bank[ident] = {"id": ident, "title": p["title"], "body": p["paragraph_text"]}
    bank = [bank[k] for k in sorted(bank, key=lambda k: stable(PROTOCOL + k))][:2000]
    used_titles, used_wordings, cases = set(excluded_titles), set(), []
    excluded_families = set(excluded_families)
    for split, row in ((s, r) for s in ("dev", "train") for r in source_rows[s]):
        if row["id"] in reserved:
            continue
        q = candidate(row)
        if not q or q["hops"] != 2 or stable(q["family"]) in excluded_families:
            continue
        titles = {title_key(p["title"]) for p in q["evidence"]}
        wording = tuple(words(q["question"]))
        if titles & used_titles or wording in used_wordings:
            continue
        possible = {}
        for p in row["paragraphs"]:
            if p["is_supporting"] or title_key(p["title"]) in titles:
                continue
            ident = stable("\x1f".join(canonical_paragraph(p)))
            d = {"id": ident, "title": p["title"], "body": p["paragraph_text"]}
            if not any(literal(a, display(d)) for a in q["answers"]):
                possible[ident] = d
        docs = list(possible.values())
        if not docs:
            continue
        order = rank(q["question"] + " " + q["anchor"], [display(d) for d in docs], limit=len(docs))
        anchored = [i for i in order if literal(q["anchor"], display(docs[i]))]
        foil = docs[(anchored or order)[0]]
        distractors = [d for d in docs if d["id"] != foil["id"] and not literal(q["anchor"], display(d))]
        cases.append({"id": q["id"], "split": split, "family": stable(q["family"]), "question": q["question"], "gold": q["evidence"][1],
                      "earlier": q["evidence"][0], "foil": foil, "anchor": q["anchor"], "answers": q["answers"],
                      "anchored_foil": bool(anchored), "distractors": distractors})
        used_titles |= titles
        used_wordings.add(wording)
        if len(cases) == count + 48:
            break
    assert len(cases) >= count, len(cases)
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "retrieval.json", {"source_sha256": digest(source), "reserved_sha256": digest(previous),
                                         "reserved_count": len(reserved), "cases": cases, "filler_bank": bank})


def load_locked_tokenizer(root):
    from transformers import AutoTokenizer

    tokenizer_path = root / "model" / "tokenizer.json"
    expected = dict((name, value) for value, name in (line.split() for line in (root / "model-sha256.txt").read_text().splitlines()))
    assert digest(tokenizer_path) == expected["tokenizer.json"]
    return AutoTokenizer.from_pretrained(root / "model", local_files_only=True), tokenizer_path


def retrieval_segments(tokenizer, row, earlier, endpoints):
    text = ("Select the passage that answers the question. Use earlier evidence to identify the right entity "
            "when it is available. Reply with A or B.\n\nQuestion: " + row["question"] +
            "\n\nEarlier passage:\n" + earlier + "\n\nOther passages:\n" + MARKER +
            "\n\nPassage A:\n" + endpoints[0] + "\n\nPassage B:\n" + endpoints[1] + "\n\nChoice:")
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True)
    return prompt.split(MARKER)


def tokenize_retrieval(tokenizer, raw, count=CASES):
    def encode(text, limit=None):
        return tokenizer.encode(text, add_special_tokens=False, **({"truncation": True, "max_length": limit} if limit else {}))

    retrieval, cases = [], []
    fill_bank = raw["filler_bank"]
    for row in raw["cases"]:
        contents = [tokenizer.decode(encode(display(row[k]), 256)) for k in ("earlier", "gold", "foil")]
        earlier, gold, foil = contents
        if not literal(row["anchor"], earlier) or not literal(row["anchor"], gold) or not any(literal(a, gold) for a in row["answers"]):
            continue
        banned = {title_key(row[k]["title"]) for k in ("earlier", "gold", "foil")}
        start = int(stable(PROTOCOL + row["id"])[:8], 16) % len(fill_bank)
        fillers, ids, length = [], [], 0
        for doc in row["distractors"] + fill_bank[start:] + fill_bank[:start]:
            d = display(doc)
            if title_key(doc["title"]) in banned or literal(row["anchor"], d) or any(literal(a, d) for a in row["answers"]):
                continue
            tokens = encode(d + "\n\n", 384)
            fillers.extend(tokens)
            ids.append(doc["id"])
            length += len(tokens)
            banned.add(title_key(doc["title"]))
            if length >= CONTEXT:
                break
        assert len(fillers) >= CONTEXT
        for expected, endpoints in enumerate(((gold, foil), (foil, gold))):
            a, b = map(encode, retrieval_segments(tokenizer, row, earlier, endpoints))
            inputs = a + fillers[:CONTEXT-len(a)-len(b)] + b
            assert len(inputs) == CONTEXT
            retrieval.append({"id": row["id"], "family": row["family"], "expected": expected, "ids": inputs})
        cases.append({"id": row["id"], "split": row["split"], "family": row["family"], "anchored_foil": row["anchored_foil"],
                      "support_ids": [row[k]["id"] for k in ("earlier", "gold")], "foil_id": row["foil"]["id"],
                      "filler_ids": ids})
        if len(cases) == count:
            break
    assert len(cases) == count
    return retrieval, cases


def prepare_inputs(root, work):
    model, revision, _ = model_identity(root)
    tokenizer, tokenizer_path = load_locked_tokenizer(root)

    def encode(text, limit=None):
        return tokenizer.encode(text, add_special_tokens=False, **({"truncation": True, "max_length": limit} if limit else {}))

    data = {"pg19": [], "retrieval": []}
    pg = work / "pg19"
    for book in json.loads((pg / "sample.json").read_text()):
        path = pg / Path(book["name"]).name
        tokens = encode(path.read_text(errors="replace"))
        block = CONTEXT + 1
        choices = range(2, len(tokens) // block - 1)
        seeds = sorted(choices, key=lambda i: stable(f"{PROTOCOL}:{path.stem}:{i}"))[:PG_WINDOWS]
        assert len(seeds) == PG_WINDOWS
        data["pg19"].append({"id": path.stem, "windows": [tokens[i*block:(i+1)*block] for i in seeds],
                              "blocks": seeds, "source_sha256": digest(path)})
    assert len(data["pg19"]) == 48

    raw = json.loads((work / "retrieval.json").read_text())
    data["retrieval"], cases = tokenize_retrieval(tokenizer, raw)
    target = work / "inputs.json"
    write_json(target, data)
    write_json(work / "manifest.json", {"protocol": PROTOCOL, "model": model, "revision": revision,
               "tokenizer_sha256": digest(tokenizer_path), "inputs_sha256": digest(target), "context": CONTEXT,
               "scored_tokens_per_window": TARGETS, "source_sha256": raw["source_sha256"],
               "reserved_sha256": raw["reserved_sha256"], "reserved_count": raw["reserved_count"],
               "pg19": [{k: r[k] for k in ("id", "blocks", "source_sha256")} for r in data["pg19"]],
               "families": Counter(row["family"] for row in cases), "retrieval": cases})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    cli = p.add_subparsers(dest="command", required=True)
    prep = cli.add_parser("freeze")
    prep.add_argument("source", type=Path)
    prep.add_argument("reserved", type=Path)
    prep.add_argument("work", type=Path)
    inputs = cli.add_parser("inputs")
    inputs.add_argument("root", type=Path)
    inputs.add_argument("work", type=Path)
    a = p.parse_args()
    if a.command == "freeze":
        freeze_retrieval(a.source, a.reserved, a.work)
    else:
        prepare_inputs(a.root, a.work)
