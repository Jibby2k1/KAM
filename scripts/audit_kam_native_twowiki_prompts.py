#!/usr/bin/env python3
"""Separate reader tokenizer normalization from evidence truncation."""

import argparse
import gzip
import json
from pathlib import Path
import sqlite3

from transformers import AutoTokenizer

from audit_kam_cross_dataset_wikipedia import tokens, wordspan
from audit_kam_natural_data import digest
from run_kam_hotpot_pool_scaling import Reader
from run_kam_native_twowiki_reader import contexts, reference_rows


def decoded(tokenizer, text, cap=None):
    truncation = {} if cap is None else {"truncation": True, "max_length": cap}
    ids = tokenizer(text, **truncation).input_ids
    return tokenizer.decode(ids, skip_special_tokens=True)


def count_matches(document, wanted):
    return sum(bool(span) and wordspan(document, span)[0] for span in wanted)


def audit(a):
    fixture = json.loads(a.fixture.read_text())
    assert fixture["source_sha256"] == digest(a.source)
    references = reference_rows(a.source, {r["id"] for r in fixture["rows"]})
    tokenizer = AutoTokenizer.from_pretrained(a.assets / "reader", local_files_only=True)
    reader = Reader.__new__(Reader)
    reader.tokenizer = tokenizer
    output = []
    db = (a.inputs / "lexical/articles.sqlite").resolve()
    with sqlite3.connect(f"{db.as_uri()}?immutable=1", uri=True) as database:
        for case in fixture["rows"]:
            arms, sentences = contexts(case, references[case["id"]], database, raw_reference=True)
            reference = list(map(tokens, sentences))
            normalized = [tokens(decoded(tokenizer, sentence)) for sentence in sentences]
            measurements = {}
            for name, documents in arms.items():
                encoded = reader.prompt({"question": case["question"], "context": documents}, [0, 1])
                clipped = tokens(decoded(tokenizer, encoded, 512).split("Evidence:", 1)[1])
                source = " ".join(title + ". " + " ".join(body) for title, body in documents)
                full = tokens(decoded(tokenizer, source))
                measurements[name] = {"raw_reference_visible": count_matches(clipped, reference),
                                      "normalized_reference_visible": count_matches(clipped, normalized),
                                      "normalized_reference_before_limits": count_matches(full, normalized)}
            output.append({"id": case["id"], "units": len(reference), "arms": measurements})
    return {"fixture_sha256": digest(a.fixture), "source_sha256": fixture["source_sha256"],
            "corpus": fixture["corpus"], "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "source", "fixture", "inputs", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    result = json.dumps(audit(args), separators=(",", ":")).encode()
    args.output.write_bytes(gzip.compress(result, mtime=0))


if __name__ == "__main__":
    main()
