#!/usr/bin/env python3
"""Replay the locked 2Wiki reader on refit article pairs."""

import argparse
import gzip
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_hotpot_pool_scaling import Reader
from run_kam_native_twowiki_reader import reference_rows
from run_kam_natural_feasibility import measure_answer
from run_kam_sentence_verifier_fullwiki import SENTENCE_BREAK


@torch.inference_mode()
def run(a):
    started = perf_counter()
    fixture = json.loads(a.fixture.read_text())
    choices = json.loads(a.picks.read_text())
    original = json.loads(gzip.decompress(a.prior.read_bytes()))
    assert choices["policy_sha256"] == digest(a.policy)
    assert fixture["source_sha256"] == digest(a.source) == original["source_sha256"]
    assert digest(a.fixture) == choices["fixture_sha256"] == original["fixture_sha256"]
    assert fixture["corpus"] == choices["corpus"] == original["corpus"]
    assert json.loads((a.assets / "revisions.json").read_text()) == choices["models"] == original["models"]
    assert choices["search_hashes"] == original["search_hashes"]
    assert (a.shard, a.shards) == (original["shard"], original["shards"])
    assert [r["id"] for r in choices["rows"]] == [q["id"] for q in fixture["rows"]]
    cases = fixture["rows"][a.shard::a.shards]
    picks = choices["rows"][a.shard::a.shards]
    references = reference_rows(a.source, {r["id"] for r in cases})
    reader = Reader(a.assets / "reader")
    database = (a.inputs / "lexical/articles.sqlite").resolve()
    metadata = json.loads((a.inputs / "lexical/completed.json").read_text())
    assert all(metadata[k] == v for k, v in fixture["corpus"].items())
    rows, prior_requests, new_requests = [], [], []
    with sqlite3.connect(f"{database.as_uri()}?immutable=1", uri=True) as db:
        for case, picked, previous in zip(cases, picks, original["rows"], strict=True):
            assert case["id"] == picked["id"] == previous["id"]
            previous_arms = previous["arms"]
            pair = picked["pairs"]["refit"]
            fixed = previous_arms["conditioned128"]
            assert picked["pairs"]["frozen"] == fixed["pair"] and picked["shortlisted"] == fixed["shortlisted"]
            answer = references[case["id"]]
            assert answer["question"] == case["question"]
            assert len(pair) == len(set(pair)) == 2

            def prompt(ids):
                docs = [db.execute("SELECT title, body FROM docs WHERE id=?", (i,)).fetchone() for i in ids]
                evidence = [(title, SENTENCE_BREAK.split(text)) for title, text in docs]
                return reader.prompt({"question": case["question"], "context": evidence}, [0, 1])

            for arm in previous_arms.values():
                prior_requests.append((prompt(arm["pair"]), answer["answer"], arm))
            row = {"id": case["id"], "frozen": {"pair": fixed["pair"], "em": fixed["em"], "f1": fixed["f1"]},
                   "refit": {"pair": pair}, "shortlisted": fixed["shortlisted"]}
            new_requests.append((prompt(pair), answer["answer"], row["refit"]))
            rows.append(row)
    old_prompts = list(dict.fromkeys(p for p, _, _ in prior_requests))
    old_outputs = dict(zip(old_prompts, reader.answers(old_prompts), strict=True))
    for prompt, answer, expected in prior_requests:
        score = measure_answer(old_outputs[prompt], answer)
        assert score["em"] == expected["em"] and score["f1"] == expected["f1"]
    new_prompts = list(dict.fromkeys(p for p, _, _ in new_requests if p not in old_outputs))
    outputs = old_outputs | dict(zip(new_prompts, reader.answers(new_prompts), strict=True))
    for prompt, answer, record in new_requests:
        score = measure_answer(outputs[prompt], answer)
        record.update({k: score[k] for k in ("em", "f1")})
    output = {k: original[k] for k in ("fixture_sha256", "models", "corpus", "search_hashes", "source_sha256")}
    return output | {"picks_sha256": digest(a.picks), "policy_sha256": digest(a.policy),
                     "prior_sha256": digest(a.prior), "shard": a.shard, "shards": a.shards,
                     "replayed_frozen_scores": len(prior_requests), "new_reader_prompts": len(new_prompts),
                     "elapsed_seconds": perf_counter() - started, "device": torch.cuda.get_device_name(), "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "fixture", "prior", "policy", "picks", "source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", required=True, type=int)
    p.add_argument("--shards", required=True, type=int)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    a.output.write_bytes(gzip.compress(json.dumps(run(a), separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
