#!/usr/bin/env python3
"""Freeze matched native MuSiQue questions with two plausible endpoint passages."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZipFile

from audit_kam_natural_data import digest, stable, words
from prepare_kam_native_musique import MEMBERS, canonical_paragraph, reserve
from prepare_kam_natural_router import title_key

REFERENCE = re.compile(r"#(\d+)")


def literal(word, text):
    return bool(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.IGNORECASE))


def candidate(row):
    steps = row["question_decomposition"]
    template = steps[-1]["question"]
    refs = {int(x) for x in REFERENCE.findall(template)}
    if len(refs) != 1 or not row["answerable"] or len(steps) not in (2, 4):
        return None
    parent, = refs
    assert 1 <= parent < len(steps)
    context = [row["paragraphs"][step["paragraph_support_idx"]] for step in steps]
    assert all(p["is_supporting"] for p in context)
    premises = [{"title": p["title"], "body": p["paragraph_text"],
                 "id": stable("\x1f".join(canonical_paragraph(p)))} for p in context]
    endpoint = premises[-1]["title"] + ". " + premises[-1]["body"]
    anchor = steps[parent-1]["answer"].strip()
    answer = row["answer"].strip()
    accepted = sorted({answer, *(x.strip() for x in row["answer_aliases"] if x.strip())})
    question = row["question"]
    visible = question + " " + " ".join(p["title"] + ". " + p["body"] for p in premises[:-1])
    if len(anchor) < 3 or len(answer) < 2 or not literal(anchor, endpoint) or literal(anchor, question):
        return None
    if not literal(anchor, premises[parent-1]["title"] + ". " + premises[parent-1]["body"]):
        return None
    if not literal(answer, endpoint) or any(literal(x, visible) for x in accepted):
        return None
    if len({p["id"] for p in premises}) != len(steps):
        return None
    family = " ".join(REFERENCE.sub("<entity>", template).casefold().strip(" ?.").split())
    return {"id": row["id"], "question": question, "hops": len(steps), "family": family,
            "bound_final": REFERENCE.sub(lambda _: anchor, template), "anchor": anchor, "answers": accepted,
            "evidence": premises}


def compatible(a, b):
    titles = [{title_key(p["title"]) for p in r["evidence"]} for r in (a, b)]
    if titles[0] & titles[1]:
        return False
    if set(map(tuple, map(words, a["answers"]))) & set(map(tuple, map(words, b["answers"]))):
        return False
    for first, other in ((a, b), (b, a)):
        all_other = other["question"] + " " + " ".join(p["title"] + " " + p["body"] for p in other["evidence"])
        if any(literal(value, all_other) for value in [first["anchor"], *first["answers"]]):
            return False
    return True


def pack(a):
    reference = json.loads(a.base_fixture.read_bytes())
    source_hash = digest(a.source)
    assert reference["source_sha256"] == source_hash
    prior_ids, prior_words = reserve([a.base_fixture, *a.reserve])
    groups, dates = defaultdict(list), sha256()
    for_book = Counter()
    with ZipFile(a.source) as archive, archive.open(MEMBERS["dev"]) as stream:
        for line in stream:
            dates.update(line)
            original = json.loads(line)
            words_ = tuple(words(original["question"]))
            if original["id"] in prior_ids or words_ in prior_words:
                continue
            option = candidate(original)
            if option:
                groups[option["hops"], option["family"]].append(option)
                for_book[str(option["hops"])] += 1
    assert dates.hexdigest() == reference["member_sha256"]["dev"]
    merged, used = [], set()
    for (hops, family), options in sorted(groups.items(), key=lambda item: stable("kam-endpoint-family-v1:" + repr(item[0]))):
        order = sorted(options, key=lambda item: stable("kam-endpoint-v1:" + item["id"]))
        for i, a_ in enumerate(order):
            titles = {title_key(p["title"]) for p in a_["evidence"]}
            if used & titles:
                continue
            for b in order[i+1:]:
                other = {title_key(p["title"]) for p in b["evidence"]}
                if not (used & other) and compatible(a_, b):
                    pair = stable("kam-endpoint-pair-v1:" + a_["id"] + "|" + b["id"])
                    merged.append({"id": pair, "hops": hops, "family": stable(family), "questions": [a_, b]})
                    used.update(titles | other)
                    break
    merged.sort(key=lambda row: row["id"])
    assert all(len(x["questions"]) == 2 for x in merged)
    info = {"source_sha256": source_hash, "dev_sha256": dates.hexdigest(),
            "reservations": {str(i): digest(p) for i, p in enumerate([a.base_fixture, *a.reserve])},
            "eligible_questions_by_depth": for_book, "pair_counts": Counter(str(x["hops"]) for x in merged)}
    return {**info, "pairs": merged}, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "base-fixture", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reserve", type=Path, action="append", default=[])
    a = p.parse_args()
    fixture, info = pack(a)
    a.output.write_bytes(gzip.compress(json.dumps(fixture, separators=(",", ":"), sort_keys=True).encode(), mtime=0))
    print(json.dumps(info, sort_keys=True))


if __name__ == "__main__":
    main()
