#!/usr/bin/env python3
"""Freeze new HotpotQA support choices and prepare common tokenized test inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from audit_kam_natural_data import rank, stable, words
from eval_kam_pretrained_endpoint_selection import literal
from kam_adapter_preservation import digest, model_identity
from kam_math_kl_protocol import write_json
from prepare_kam_natural_router import title_key

PROTOCOL = "kam-live-token-transfer-v1"
PER_COHORT = 96
PG_WINDOWS = 8
SCORE_TOKENS = 512


def hotpot(sources, reservations, work):
    import pyarrow.parquet as pq

    reserved = set(reservations.read_text().split())
    eligible = []
    for source in sources:
        for groups in pq.ParquetFile(source).iter_batches(batch_size=1024):
            for row in groups.to_pylist():
                answer = row["answer"].strip()
                if row["id"] in reserved or row["type"] != "bridge" or row["level"] != "hard" or len(answer) < 2 or answer.casefold() in ("yes", "no"):
                    continue
                context = dict(zip(row["context"]["title"], row["context"]["sentences"], strict=True))
                facts = list(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"], strict=True))
                titles = set(t for t, _ in facts)
                if len(titles) != 2 or not all(t in context and 0 <= i < len(context[t]) for t, i in facts):
                    continue
                endpoints = {t for t, i in facts if literal(answer, context[t][i])}
                if len(endpoints) != 1:
                    continue
                gold, = endpoints
                first, = titles - endpoints
                if literal(answer, " ".join(context[first])) or len(words(gold)) < 2:
                    continue
                if not literal(gold, " ".join(context[first])):
                    continue
                alternatives = [t for t in sorted(set(context) - titles) if context[t] and not literal(answer, " ".join(context[t]))]
                if not alternatives:
                    continue
                ordered = rank(row["question"] + " " + gold,
                               [t + " " + " ".join(context[t]) for t in alternatives], limit=len(alternatives))
                links = [i for i in ordered if len(words(alternatives[i])) >= 2 and literal(alternatives[i], " ".join(context[first]))]
                foil = alternatives[(links or ordered)[0]]
                snippets = [t + ". " + " ".join(context[t]) for t in (first, gold, foil)]
                eligible.append({"id": row["id"], "question": row["question"], "answer": answer, "titles": [first, gold, foil],
                                 "cohort": "linked" if links else "unlinked", "passages": snippets})
    rows, used_titles, seen_questions, counts = [], set(), set(), Counter()
    for row in sorted(eligible, key=lambda item: stable(PROTOCOL + item["id"])):
        titles = set(map(title_key, row["titles"]))
        wording = tuple(words(row["question"]))
        category = row["cohort"]
        if counts[category] == PER_COHORT or titles & used_titles or wording in seen_questions:
            continue
        rows.append(row)
        counts[category] += 1
        used_titles |= titles
        seen_questions.add(wording)
        if all(counts[c] == PER_COHORT for c in ("linked", "unlinked")):
            break
    assert counts["linked"] >= 24 and counts["unlinked"] == PER_COHORT, counts
    work.mkdir(parents=True, exist_ok=True)
    file = work / "retrieval.json"
    write_json(file, rows)
    write_json(work / "data-manifest.json", {"protocol": PROTOCOL, "retrieval_sha256": digest(file),
               "source_sha256": [digest(s) for s in sources], "reserved_sha256": digest(reservations),
               "reserved_count": len(reserved), "counts": counts,
               "cases": [{"id": r["id"], "cohort": r["cohort"], "titles": [stable(title_key(t)) for t in r["titles"]]} for r in rows]})


def prepare_inputs(root, work):
    from transformers import AutoTokenizer

    config = json.loads((work / "data-manifest.json").read_text())
    assert config["protocol"] == PROTOCOL
    model, revision, _ = model_identity(root)
    tokenizer_path = root / "model" / "tokenizer.json"
    expected = dict((name, value) for value, name in (line.split() for line in (root / "model-sha256.txt").read_text().splitlines()))
    assert digest(tokenizer_path) == expected["tokenizer.json"]
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
    text_file = work / "retrieval.json"
    assert digest(text_file) == config["retrieval_sha256"]
    retrieval = []
    for raw in json.loads(text_file.read_text()):
        texts = [tokenizer.decode(tokenizer.encode(p, add_special_tokens=False, truncation=True, max_length=256)) for p in raw["passages"]]
        if not literal(raw["titles"][1], texts[0]) or not literal(raw["answer"], texts[1]):
            continue
        for slot in (0, 1):
            a, b = (texts[1], texts[2]) if slot == 0 else (texts[2], texts[1])
            prompt = ("Select the passage that answers the question. Use earlier evidence to identify the right entity "
                      "when it is available. Reply with A or B.\n\nQuestion: " + raw["question"] +
                      "\n\nEarlier passage:\n" + texts[0] + "\n\nPassage A:\n" + a + "\n\nPassage B:\n" + b + "\n\nChoice:")
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                   add_generation_prompt=True)
            retrieval.append({"id": raw["id"], "cohort": raw["cohort"], "expected": slot,
                              "ids": tokenizer.encode(prompt, add_special_tokens=False)})
    pg = work / "pg19"
    books = json.loads((pg / "sample.json").read_text())
    prepared = {"pg19": [], "retrieval": retrieval}
    raw_hashes = {}
    for book in books:
        path = pg / Path(book["name"]).name
        raw_hashes[path.stem] = digest(path)
        tokens = tokenizer.encode(path.read_text(errors="replace"), add_special_tokens=False)
        block = SCORE_TOKENS + 1
        choices = range(8, len(tokens) // block - 1)
        selected = sorted(choices, key=lambda i: stable(f"{PROTOCOL}:{path.stem}:{i}"))[:PG_WINDOWS]
        assert len(selected) == PG_WINDOWS
        prepared["pg19"].append({"id": path.stem, "windows": [tokens[i*block:(i+1)*block] for i in selected],
                                 "blocks": selected})
    assert len(prepared["pg19"]) == 24 and len(retrieval) >= 2 * 100
    target = work / "inputs.json"
    write_json(target, prepared)
    write_json(work / "input-manifest.json", {"protocol": PROTOCOL, "model": model, "revision": revision,
               "data_manifest_sha256": digest(work / "data-manifest.json"), "tokenizer_sha256": digest(tokenizer_path),
               "inputs_sha256": digest(target), "pg19": [{"id": r["id"], "blocks": r["blocks"], "sha256": raw_hashes[r["id"]]}
                                                              for r in prepared["pg19"]],
               "retrieval": [{k: r[k] for k in ("id", "cohort", "expected")} | {"tokens": len(r["ids"])} for r in retrieval]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    source = sub.add_parser("hotpot")
    source.add_argument("work", type=Path)
    source.add_argument("reservations", type=Path)
    source.add_argument("source", nargs="+", type=Path)
    inputs = sub.add_parser("inputs")
    inputs.add_argument("root", type=Path)
    inputs.add_argument("work", type=Path)
    args = p.parse_args()
    if args.command == "hotpot":
        hotpot(args.source, args.reservations, args.work)
    else:
        prepare_inputs(args.root, args.work)
