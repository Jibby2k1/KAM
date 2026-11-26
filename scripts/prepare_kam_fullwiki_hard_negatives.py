#!/usr/bin/env python3
"""Freeze document-disjoint fit and test questions for the hard-negative comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from audit_kam_natural_data import digest, stable, words
from prepare_kam_natural_router import title_key


def freeze(a):
    manifest = json.loads((a.existing / "manifest.json").read_text())
    native = json.loads((a.existing / "lexical.json").read_text())
    assert manifest["validation_sha256"] == digest(a.validation)
    assert manifest["source_sha256"] == digest(a.corpus)
    assert native["mask_sha256"] == digest(a.mask)
    prior = a.existing / "queries.json"
    assert manifest["queries_sha256"] == digest(prior)
    existing = [q for group in json.loads(prior.read_text()).values() for q in group]
    for path in a.reserve_fixture:
        reserve = json.loads(path.read_text())
        assert reserve["source_sha256"] == manifest["source_sha256"]
        existing.extend({**q, "_id": q["id"]} for group in reserve["rows"].values() for q in group)
    seen = {" ".join(words(q["question"])) for q in existing}
    reserved = {q["_id"] for q in existing}
    if a.prior_text_source:
        prior_questions = json.loads(a.prior_text_source.read_text())
        seen.update(" ".join(words(q["question"])) for q in prior_questions)
        reserved.update(q["_id"] for q in prior_questions)
    banned = a.existing / "excluded_titles.txt"
    assert manifest["excluded_titles_sha256"] == digest(banned)
    titles = set(banned.read_text().splitlines()) | {t for q in existing for t in q["gold_titles"]}
    inputs = a.external_test_source or [a.validation]
    columns = ["id", "question", "answer", "type", "supporting_facts"]
    rows = [row for path in inputs for row in pq.read_table(path, columns=columns).to_pylist()]
    assert len({row["id"] for row in rows}) == len(rows)
    eligible = []
    for row in rows:
        if row["type"] != "bridge" or row["id"] in reserved:
            continue
        gold = sorted({title_key(t) for t in row["supporting_facts"]["title"]})
        if len(gold) == 2:
            eligible.append({"id": row["id"], "question": row["question"], "answer": row["answer"], "gold_titles": gold})
    namespace = a.namespace + ":" if a.namespace else (
        "kam-link-strength-v1:" if a.test_only else "kam-hard-negative-v1:"
    )
    eligible.sort(key=lambda row: stable(namespace + row["id"]))
    # Reserve both partitions before checking coverage or running any experiment.
    groups = {}
    cohorts = (("test", a.test_count),) if a.test_only else (("training", a.training_count), ("test", a.test_count))
    for group, count in cohorts:
        chosen = []
        for case in eligible:
            tokens = " ".join(words(case["question"]))
            gold = set(case["gold_titles"])
            if gold & titles or tokens in seen:
                continue
            chosen.append(case)
            seen.add(tokens)
            titles.update(gold)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Requested {count} {group} cases, found {len(chosen)} title-disjoint cases")
        groups[group] = chosen
    wanted = {t for group in groups.values() for row in group for t in row["gold_titles"]}
    rows_by_title = {}
    mask = np.load(a.mask, allow_pickle=False)
    base = 0
    for batch in pq.ParquetFile(a.corpus).iter_batches(batch_size=65536, columns=["title"]):
        for i, title in enumerate(batch.column(0).to_pylist(), base):
            t = title_key(title) if title else ""
            if t in wanted and mask[i]:
                assert t not in rows_by_title
                rows_by_title[t] = i
        base += len(batch)
    assert base == len(mask)
    for group in groups.values():
        for row in group:
            row["gold_ids"] = [rows_by_title.get(title) for title in row["gold_titles"]]
    output = {"source_sha256": manifest["source_sha256"], "validation_sha256": manifest["validation_sha256"],
              "previous_queries_sha256": manifest["queries_sha256"], "training_depth": 64, "test_depth": a.test_depth,
              "primary_metric": "complete support on eligible test pools", "rows": groups}
    if len(a.reserve_fixture) == 1:
        output["reserved_fixture_sha256"] = digest(a.reserve_fixture[0])
    elif a.reserve_fixture:
        output["reserved_fixtures_sha256"] = [digest(path) for path in a.reserve_fixture]
    if a.external_test_source:
        output["external_test_sha256"] = [digest(path) for path in a.external_test_source]
    if a.prior_text_source:
        output["prior_text_source_sha256"] = digest(a.prior_text_source)
    a.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("validation", "existing", "corpus", "mask", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reserve-fixture", type=Path, action="append", default=[])
    p.add_argument("--external-test-source", type=Path, action="append", default=[])
    p.add_argument("--prior-text-source", type=Path)
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--test-count", type=int, default=192)
    p.add_argument("--training-count", type=int, default=256)
    p.add_argument("--test-depth", type=int, default=256)
    p.add_argument("--namespace")
    args = p.parse_args()
    if min(args.test_count, args.training_count) < 1:
        p.error("cohort sizes must be positive")
    if args.test_depth < 2:
        p.error("--test-depth must be at least two")
    if args.external_test_source and not args.prior_text_source:
        p.error("--external-test-source requires --prior-text-source")
    freeze(args)


if __name__ == "__main__":
    main()
