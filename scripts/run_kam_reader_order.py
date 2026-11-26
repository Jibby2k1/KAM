#!/usr/bin/env python3
"""Replay both document orders on a fixed held-out evidence pair."""

import argparse
from contextlib import closing
import gzip
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import measure_answer
from run_kam_sentence_verifier_fullwiki import SENTENCE_BREAK


@torch.inference_mode()
def run(a):
    start = perf_counter()
    fixture = json.loads(a.fixture.read_bytes())
    ref = json.loads(gzip.decompress(a.reference.read_bytes()))
    models = json.loads((a.assets / "revisions.json").read_bytes())
    provenance = {"fixture_sha256": digest(a.fixture), "models": models,
                  "index_sha256": digest(a.inputs / "lexical" / "completed.json")}
    assert all(all(s[k] == v for k, v in provenance.items()) for s in ref)
    chosen = {r["id"]: r["arms"]["conditioned_union"]["pair"] for s in ref for r in s["rows"]}
    questions = fixture["rows"]["test"]
    assert len(chosen) == len(questions) == 1024 and chosen.keys() == {q["id"] for q in questions}
    reader = Reader(a.assets / "reader")
    rows, pending = [], {}
    path = a.inputs / "lexical" / "articles.sqlite"
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?immutable=1", uri=True)) as db:
        db.execute("PRAGMA mmap_size=1073741824")
        for case in questions[a.shard::a.shards]:
            ids = chosen[case["id"]]
            docs = {i: (title, SENTENCE_BREAK.split(body)) for i, title, body in db.execute(
                "SELECT id,title,body FROM docs WHERE id IN (?,?)", ids)}
            assert len(ids) == len(docs) == 2
            evidence = {"question": case["question"], "context": [docs[j] for j in ids]}
            row = {"id": case["id"], "pair": ids, "exact": {}}
            for name, order in (("reference", (0, 1)), ("reversed", (1, 0))):
                prompt = reader.prompt(evidence, order)
                pending.setdefault(prompt, []).append((row["exact"], name, case["answer"]))
            rows.append(row)

    for answer, targets in zip(reader.answers(list(pending)), pending.values(), strict=True):
        for result, name, expected in targets:
            result[name] = bool(measure_answer(answer, expected)["em"])
    return {**provenance, "reference_sha256": digest(a.reference), "gpu": torch.cuda.get_device_name(),
            "shard": a.shard, "shards": a.shards, "seconds": perf_counter() - start, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "reference", "inputs", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    a = parser.parse_args()
    assert 0 <= a.shard < a.shards
    payload = json.dumps(run(a), allow_nan=False, separators=(",", ":")).encode()
    a.output.write_bytes(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    main()
