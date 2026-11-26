#!/usr/bin/env python3
"""Extract verified passage and supporting-sentence references for a frozen HotpotQA cohort."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from audit_kam_natural_data import digest
from prepare_kam_natural_router import title_key


def extract(fixture_path, parts):
    fixture = json.loads(fixture_path.read_bytes())
    cohort = {q["id"]: q for q in fixture["rows"]["test"]}
    assert len(cohort) == 1024
    hashes = [digest(p) for p in parts]
    assert hashes == fixture["external_test_sha256"]
    found = {}
    columns = ["id", "question", "answer", "type", "context", "supporting_facts"]
    for p in parts:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1024, columns=columns):
            for r in batch.to_pylist():
                if r["id"] not in cohort:
                    continue
                f = cohort[r["id"]]
                assert r["id"] not in found and r["type"] == "bridge"
                assert r["question"] == f["question"] and r["answer"] == f["answer"]
                ctx = {}
                for title, passage in zip(r["context"]["title"], r["context"]["sentences"], strict=True):
                    ctx.setdefault(title_key(title), []).append((title, passage))
                evidence = {}
                facts = r["supporting_facts"]
                for title, i in zip(facts["title"], facts["sent_id"], strict=True):
                    evidence.setdefault(title_key(title), set()).add(i)
                assert set(evidence) == set(f["gold_titles"]) and len(evidence) == 2
                assert len(f["gold_ids"]) == len(set(f["gold_ids"])) == 2 and None not in f["gold_ids"]
                articles = []
                for title, doc in zip(f["gold_titles"], f["gold_ids"], strict=True):
                    sources = ctx[title]
                    assert len(sources) == 1
                    source_title, passage = sources[0]
                    positions = sorted(evidence[title])
                    valid = [i for i in positions if 0 <= i < len(passage) and passage[i].strip()]
                    invalid = [i for i in positions if i not in valid]
                    articles.append({"title": source_title, "row_id": doc, "source_passage": passage,
                                     "support_positions": valid, "invalid_support_positions": invalid})
                found[r["id"]] = {"id": r["id"], "articles": articles}
    assert found.keys() == cohort.keys()
    return {"fixture_sha256": digest(fixture_path), "dataset_sha256": hashes,
            "rows": [found[q["id"]] for q in fixture["rows"]["test"]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--part", type=Path, required=True, action="append")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(extract(a.fixture, a.part), ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
