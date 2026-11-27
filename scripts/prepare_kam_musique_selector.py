#!/usr/bin/env python3
"""Seal selector training questions disjoint from decomposer fit and final test."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from audit_kam_natural_data import digest, words
from prepare_kam_musique_decomposition import title_sha
from prepare_kam_native_musique import MEMBERS, canonical_paragraph
from run_kam_native_musique import corpus_and_fixture


def prepare(a, bank, base):
    data = json.loads(gzip.decompress(a.training.read_bytes()))
    train = data["splits"]["fit"]
    valid = data["splits"]["validation"]
    assert data["fixture_sha256"] == digest(a.base_fixture) and digest(a.source) == base["source_sha256"]
    parts = {"decomposer_fit": train, "selector_fit": valid, "test": base["rows"]}
    title_sets = {name: {t for r in rows for t in r["support_title_sha256"]} for name, rows in parts.items()
                  if name != "test"}
    title_sets["test"] = {title_sha(bank[i]["title"]) for q in parts["test"] for i in q["gold"]}
    assert all(not (x & y) for k, x in title_sets.items() for o, y in title_sets.items() if k != o)
    question_sets = {name: {tuple(words(r["question"])) for r in rows} for name, rows in parts.items()}
    assert all(not (x & y) for k, x in question_sets.items() for o, y in question_sets.items() if k != o)
    text_keys = {canonical_paragraph({"title": p["title"], "paragraph_text": p["body"]}): p["id"] for p in bank}
    exact = {r["id"]: r for r in valid}
    output = {}
    member = sha256()
    with ZipFile(a.source) as archive, archive.open(MEMBERS["train"]) as stream:
        for line in stream:
            member.update(line)
            row = json.loads(line)
            id_ = row["id"]
            if id_ not in exact:
                continue
            assert row["question"] == exact[id_]["question"]
            evidence = [p for p in row["paragraphs"] if p["is_supporting"]]
            gold = sorted({text_keys[canonical_paragraph(p)] for p in evidence})
            assert len(gold) == len(exact[id_]["steps"])
            assert {title_sha(p["title"]) for p in evidence} == set(exact[id_]["support_title_sha256"])
            output[id_] = {"id": id_, "question": row["question"], "hops": len(gold), "gold": gold}
    assert member.hexdigest() == base["member_sha256"]["train"] and output.keys() == exact.keys()
    rows = [output[q["id"]] for q in valid]
    return {"base_fixture_sha256": digest(a.base_fixture), "source_sha256": base["source_sha256"],
            "corpus_sha256": base["corpus_sha256"], "documents": base["documents"],
            "decomposer_data_sha256": digest(a.training), "support_title_count": {k: len(v) for k, v in title_sets.items()},
            "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "base-fixture", "training", "source", "output", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.fixture = a.base_fixture
    bank, base = corpus_and_fixture(a)
    output = prepare(a, bank, base)
    a.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
    keys = dict(output)
    keys["rows"] = [{k: v for k, v in r.items() if k != "question"} for r in output["rows"]]
    keys["fixture_sha256"] = digest(a.output)
    a.manifest.write_bytes(gzip.compress(json.dumps(keys, separators=(",", ":")).encode(), mtime=0))
    print(Counter(r["hops"] for r in output["rows"]))


if __name__ == "__main__":
    main()
