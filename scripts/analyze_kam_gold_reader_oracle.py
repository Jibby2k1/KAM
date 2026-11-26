#!/usr/bin/env python3
"""Audit frozen oracle inputs and report paired reader limits."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_sentence_verifier import paired

ARMS = ("indexed_gold_articles", "dataset_gold_passages", "annotated_gold_sentences")


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def summarize(rows):
    scored = {arm: [r["exact"][arm] for r in rows if r["exact"][arm] is not None] for arm in ARMS}
    contrasts = {}
    for a, b in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[1]), (ARMS[2], ARMS[0])):
        pairs = [(r["exact"][a], r["exact"][b]) for r in rows if None not in (r["exact"][a], r["exact"][b])]
        if pairs:
            x, y = map(list, zip(*pairs))
            contrasts[a + "_vs_" + b] = {"count": len(pairs), "first_exact": sum(x), "second_exact": sum(y),
                                           **paired(x, y)}
    return {"questions": len(rows), "scores": {k: {"count": len(v), "exact": sum(v)} for k, v in scored.items()},
            "contrasts": contrasts}


def evaluate(shards, fixture_path, evidence_path):
    fx, ev = (gzip.decompress(p.read_bytes()) for p in (fixture_path, evidence_path))
    fixture, evidence = json.loads(fx), json.loads(ev)
    digest = lambda b: sha256(b).hexdigest()
    assert evidence["fixture_sha256"] == digest(fx)
    assert evidence["dataset_sha256"] == fixture["external_test_sha256"]
    fixture_rows = {q["id"]: q for q in fixture["rows"]["test"]}
    evidence_rows = {q["id"]: q["articles"] for q in evidence["rows"]}
    assert evidence_rows.keys() == fixture_rows.keys() and len(fixture_rows) == 1024
    assert len(shards) == 4 and sorted(s["shard"] for s in shards) == list(range(4))
    assert all(s["fixture_sha256"] == digest(fx) and s["evidence_sha256"] == digest(ev) for s in shards)
    fixed = {k: shards[0][k] for k in ("index_sha256", "models")}
    assert all(all(s[k] == v for k, v in fixed.items()) for s in shards)
    rows = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == 1024 and {r["id"] for r in rows} == fixture_rows.keys()
    for r in rows:
        articles = evidence_rows[r["id"]]
        assert [d["row_id"] for d in articles] == fixture_rows[r["id"]]["gold_ids"]
        n, invalid = (sum(len(d[name]) for d in articles) for name in ("support_positions", "invalid_support_positions"))
        v = r["visibility"]
        assert v["annotated_valid"] == n and v["invalid_annotations"] == invalid
        assert 0 <= v["in_reader_prefix"] <= v["in_article"] <= n
        assert set(r["exact"]) == set(ARMS)
        assert all(isinstance(r["exact"][arm], bool) for arm in ARMS[:2])
        assert (r["exact"][ARMS[2]] is None) == bool(invalid)
    valid = [r for r in rows if not r["visibility"]["invalid_annotations"]]
    groups = {
        "all": rows,
        "valid_annotations": valid,
        "all_support_visible_in_reader_prefix": [
            r for r in valid if r["visibility"]["in_reader_prefix"] == r["visibility"]["annotated_valid"]],
        "some_support_missing_in_reader_prefix": [
            r for r in valid if r["visibility"]["in_reader_prefix"] < r["visibility"]["annotated_valid"]],
    }
    return {**fixed, "fixture_sha256": digest(fx), "evidence_sha256": digest(ev),
            "gpus": dict(Counter(s["gpu"] for s in shards)),
            "annotation_counts": {k: sum(r["visibility"][k] for r in rows) for k in rows[0]["visibility"]},
            "questions_with_invalid_annotation": len(rows) - len(valid),
            "cohorts": {k: summarize(cases) for k, cases in groups.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    for name in ("fixture", "evidence", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--write-replay", type=Path)
    a = p.parse_args()
    shards = ([read(x) for x in sorted(a.input.glob("reader-*.json.gz"))] if a.input.is_dir() else read(a.input))
    result = evaluate(shards, a.fixture, a.evidence)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        a.write_replay.write_bytes(gzip.compress(json.dumps(shards, separators=(",", ":"), allow_nan=False).encode(),
                                                 mtime=0))


if __name__ == "__main__":
    main()
