#!/usr/bin/env python3
"""Freeze an unpadded, full-article panel from the official QuALITY dev release."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import zipfile

from audit_kam_natural_data import stable
from kam_adapter_preservation import digest
from kam_math_kl_protocol import write_json
from prepare_kam_long_sparse import load_locked_tokenizer

PROTOCOL = "kam-quality-native-v1"
ENTRY = "QuALITY.v1.0.1.htmlstripped.dev"
LABELS = ("A", "B", "C", "D")
REPO_REVISION = "05e85750d4c5444d2a0a4ad299f6df5f4df06068"
ARTICLE_CAP = 4
MAX_LENGTH = 8192
MIN_LENGTH = 1024


def render(tokenizer, article, question, options, order):
    text = "Read the article and choose the answer to the question. Reply with A, B, C, or D.\n\n"
    text += f"Article:\n{article}\n\nQuestion: {question}\n\n"
    text += "\n".join(f"{label}. {options[i]}" for label,i in zip(LABELS, order, strict=True))
    text += "\n\nChoice:"
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(prompt, add_special_tokens=False, verbose=False)


def prepare(root, source, work):
    tokenizer, tok_path = load_locked_tokenizer(root)
    with zipfile.ZipFile(source) as release:
        members = [name for name in release.namelist() if Path(name).name == ENTRY and not name.startswith("__MACOSX/")]
        filename, = members
        data = [json.loads(line) for line in release.read(filename).splitlines() if line.strip()]
    articles = defaultdict(list)
    previous = {}
    question_ids = set()
    rejects = Counter()
    for row in data:
        aid, article = row["article_id"], row["article"].strip()
        assert previous.setdefault(aid, article) == article
        for record in row["questions"]:
            qid = record["question_unique_id"]
            assert qid not in question_ids
            question_ids.add(qid)
            q, options = record["question"], [opt.strip() for opt in record["options"]]
            answer = int(record["gold_label"])-1
            hard = int(record["difficult"])
            assert 0 <= answer < 4 and hard in (0,1) and len(options) == 4
            if len(set(options)) != 4 or not all(options):
                rejects["invalid_options"] += 1
                continue
            orders = [[(i + offset) % 4 for i in range(4)] for offset in range(4)]
            # Measure fit using the complete article and every answer placement.
            first = render(tokenizer, article, q, options, orders[0])
            if not MIN_LENGTH <= len(first) <= MAX_LENGTH:
                rejects["width_initial"] += 1
                continue
            article_id = str(aid)
            articles[article_id].append({"id": qid, "article_id": article_id, "hard": hard,
                  "answer": answer, "question": q, "options": options, "article": article, "orders": orders, "first": first})
    selected, panel = [], []
    for aid in sorted(articles):
        table = articles[aid]
        eligible = {0: [], 1: []}
        for row in sorted(table, key=lambda r: stable(PROTOCOL+r["id"])):
            cohort = row["hard"]
            if len(eligible[cohort]) >= ARTICLE_CAP:
                continue
            q, opt, art = row["question"], row["options"], row["article"]
            rtl = [row["first"]] + [render(tokenizer, art, q, opt, order) for order in row["orders"][1:]]
            if not all(MIN_LENGTH <= len(ids) <= MAX_LENGTH for ids in rtl):
                rejects["width_rotation"] += 1
                continue
            short = [render(tokenizer, "", q, opt, order) for order in row["orders"]]
            record = {k: row[k] for k in ("id","article_id","hard","answer","orders")}
            eligible[cohort].append((record | {"lengths": [len(ids) for ids in rtl]},
                                      {"id": row["id"], "full": rtl, "question_only": short}))
        both = eligible[0][:ARTICLE_CAP//2] + eligible[1][:ARTICLE_CAP//2]
        leftovers = eligible[0][ARTICLE_CAP//2:] + eligible[1][ARTICLE_CAP//2:]
        both += sorted(leftovers, key=lambda r: stable(PROTOCOL+r[0]["id"]))[:ARTICLE_CAP-len(both)]
        for r, tokens in both:
            selected.append(r)
            panel.append(tokens)
    assert len(selected) >= 100 and len({r["article_id"] for r in selected}) >= 25
    assert len([r for r in selected if r["hard"]]) >= 40
    path = work / "quality-inputs.json"
    write_json(path, panel)
    write_json(work / "quality-manifest.json", {"protocol": PROTOCOL, "publisher_revision": REPO_REVISION,
        "publisher_sha256": digest(source), "entry": filename, "tokenizer_sha256": digest(tok_path),
        "inputs_sha256": digest(path), "source_article_count": len(previous), "source_question_count": len(question_ids),
        "width_rejects": rejects, "article_cap": ARTICLE_CAP, "selected": selected})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root","source","work"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    prepare(args.root, args.source, args.work)
