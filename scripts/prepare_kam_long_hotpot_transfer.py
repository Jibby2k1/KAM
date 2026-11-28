#!/usr/bin/env python3
"""Freeze a native HotpotQA comparison with linked and unlinked distractors."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

from audit_kam_natural_data import rank, stable, words
from kam_adapter_preservation import digest
from kam_math_kl_protocol import audit, write_json
from prepare_kam_long_sparse import CONTEXT, display, literal, load_locked_tokenizer, tokenize_retrieval
from prepare_kam_natural_router import title_key

PROTOCOL = "kam-long-hotpot-transfer-v1"
PER_COHORT = 64
WIDTHS = (CONTEXT, 2*CONTEXT)
IDENTITY = re.compile(rb"(?<![a-f0-9])[a-f0-9]{24}(?![a-f0-9])")


def read(path):
    return json.loads(path.read_text())


def reservations(results):
    reserved = set()
    suffixes = {".json", ".jsonl", ".tsv", ".csv", ".gz"}
    for path in results.rglob("*"):
        if path.suffix not in suffixes or not path.is_file():
            continue
        open_data = gzip.open if path.suffix == ".gz" else open
        with open_data(path, "rb") as source:
            for block in source:
                reserved.update(m.group() for m in IDENTITY.finditer(block))
    return {x.decode() for x in reserved}


def documents(row):
    result = {}
    for title, sentences in row["context"]:
        key = " ".join(sentences).strip()
        if key:
            result[title] = {"title": title, "body": key, "id": stable(title_key(title) + "\x1f" + " ".join(words(key)))}
    return result


def candidate(row, docs, split):
    if row["type"] != "bridge" or row["level"] != "hard":
        return None
    answer = row["answer"].strip()
    if len(answer) < 2 or answer.casefold() in ("yes", "no"):
        return None
    support = {title for title, _ in row["supporting_facts"]}
    if len(support) != 2 or not support <= docs.keys():
        return None
    lines = dict(row["context"])
    if any(not 0 <= i < len(lines[title]) for title, i in row["supporting_facts"]):
        return None
    hits = {title for title, i in row["supporting_facts"] if literal(answer, lines[title][i])}
    if len(hits) != 1:
        return None
    gold, = hits
    earlier, = support-{gold}
    if len(words(gold)) < 2 or not literal(gold, docs[earlier]["body"]) or literal(gold, row["question"]):
        return None
    if literal(answer, display(docs[earlier])) or literal(answer, row["question"]):
        return None
    alternatives = [d for t, d in docs.items() if t not in support and not literal(answer, display(d))]
    if not alternatives:
        return None
    order = rank(row["question"]+" "+gold, [display(d) for d in alternatives], limit=len(alternatives))
    linked = [i for i in order if len(words(alternatives[i]["title"])) > 1
              and literal(alternatives[i]["title"], docs[earlier]["body"])]
    foil = alternatives[(linked or order)[0]]
    return {"id": row["_id"], "question": row["question"], "answers": [answer], "anchor": gold,
            "family": stable(row["_id"]), "split": split, "earlier": docs[earlier], "gold": docs[gold], "foil": foil,
            "anchored_foil": bool(linked), "distractors": [d for d in alternatives if d != foil]}


def dataset(source):
    if source.suffix != ".parquet":
        return read(source)
    import pyarrow.parquet as pq

    records = []
    for batch in pq.ParquetFile(source).iter_batches(batch_size=512):
        for r in batch.to_pylist():
            context, facts = r["context"], r["supporting_facts"]
            records.append(r | {"_id": r["id"], "context": list(zip(context["title"], context["sentences"], strict=True)),
                                "supporting_facts": list(zip(facts["title"], facts["sent_id"], strict=True))})
    return records


def freeze(source, results, work):
    source_rows = dataset(source)
    split = "train" if source.suffix == ".parquet" else "dev"
    cached_reserve = work / "reservations.txt"
    if cached_reserve.exists():
        reserved = set(cached_reserve.read_text().split())
    else:
        reserved = reservations(results)
        cached_reserve.write_text("\n".join(sorted(reserved))+"\n")
    bank, cases, used = {}, [], set()
    counts = Counter()
    for row in sorted(source_rows, key=lambda r: stable(PROTOCOL+r["_id"])):
        docs = documents(row)
        support = {t for t, _ in row["supporting_facts"]}
        for title, doc in docs.items():
            if title not in support:
                bank[doc["id"]] = doc
        if row["_id"] in reserved:
            continue
        c = candidate(row, docs, split)
        if c is None:
            continue
        kind = "linked" if c["anchored_foil"] else "unlinked"
        titles = {title_key(c[key]["title"]) for key in ("earlier", "gold")}
        if titles & used or counts[kind] == PER_COHORT + 24:
            continue
        cases.append(c)
        used |= titles
        counts[kind] += 1
    print("reserved", len(reserved), "source split", split, "eligible", dict(counts))
    assert all(counts[kind] >= PER_COHORT for kind in ("linked", "unlinked")), counts
    filler = [bank[key] for key in sorted(bank, key=lambda i: stable(PROTOCOL+i))[:3000]]
    write_json(work / "native.json", {"source_sha256": digest(source), "reservations_sha256": digest(cached_reserve),
        "reserved_count": len(reserved), "source_split": split, "cases": cases, "filler_bank": filler})


def encode(root, work):
    raw = read(work / "native.json")
    tokenizer, tokenizer_path = load_locked_tokenizer(root)
    fixture = {}
    for width in WIDTHS:
        all_prompts, all_cases = [], []
        for linked in (True, False):
            cohort = [r for r in raw["cases"] if r["anchored_foil"] == linked]
            prompts, cases = tokenize_retrieval(tokenizer, raw | {"cases": cohort}, PER_COHORT, context=width)
            all_prompts.extend(prompts)
            all_cases.extend(cases)
        path = work / f"hotpot-{width}.json"
        write_json(path, all_prompts)
        fixture[width] = {"sha256": digest(path), "questions": all_cases}
    assert [r["id"] for r in fixture[CONTEXT]["questions"]] == [r["id"] for r in fixture[2*CONTEXT]["questions"]]
    write_json(work / "hotpot-manifest.json", {"protocol": PROTOCOL, "tokenizer_sha256": digest(tokenizer_path),
        "source_sha256": raw["source_sha256"], "reservations_sha256": raw["reservations_sha256"],
        "reserved_count": raw["reserved_count"], "source_split": raw["source_split"], "fixture": fixture})
    selected = {r["id"] for r in fixture[CONTEXT]["questions"]}
    docs = {r[side]["id"]: r[side]["body"] for r in raw["cases"] if r["id"] in selected
            for side in ("earlier", "gold", "foil")}
    audit_dir = work / "audit"
    audit_dir.mkdir(exist_ok=True)
    write_json(audit_dir/"dev.json", [])
    write_json(audit_dir/"test.json", [{"id": k, "problem": v} for k, v in sorted(docs.items())])
    audit(root, audit_dir, work/"source-audit.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    cli = p.add_subparsers(dest="mode", required=True)
    f = cli.add_parser("freeze")
    for name in ("source", "results", "work"):
        f.add_argument(name, type=Path)
    e = cli.add_parser("encode")
    for name in ("root", "work"):
        e.add_argument(name, type=Path)
    r = cli.add_parser("reserve")
    r.add_argument("results", type=Path)
    r.add_argument("output", type=Path)
    a = p.parse_args()
    if a.mode == "freeze":
        a.work.mkdir(parents=True, exist_ok=True)
        freeze(a.source, a.results, a.work)
    elif a.mode == "reserve":
        a.output.write_text("\n".join(sorted(reservations(a.results)))+"\n")
    else:
        encode(a.root, a.work)
