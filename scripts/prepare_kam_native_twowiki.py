#!/usr/bin/env python3
"""Freeze title-disjoint native 2Wiki pairs matched by type and annotation count."""

import argparse
from collections import defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

import pyarrow.parquet as pq

from audit_kam_natural_data import digest, stable, words
from prepare_kam_natural_router import title_key

QUOTAS = {"comparison": 192, "compositional": 192, "inference": 128}
FIT_QUOTAS = {"comparison": 112, "compositional": 128, "inference": 16}


def build(source, coverage_path, reservations, holdout=None):
    coverage = json.loads(gzip.decompress(coverage_path.read_bytes()))
    assert coverage["data_sha256"]["twowiki"] == digest(source)
    raw = pq.read_table(source, columns=["id", "question"]).to_pylist()
    wording = {r["id"]: r["question"] for r in raw}
    assert len(wording) == len(raw)
    held, used_titles, used_questions = set(), set(), set()
    for path in reservations:
        for row in json.loads(path.read_text())["rows"]:
            held.add(row["id"])
            used_questions.add(tuple(words(row["question"])))
            used_titles.update(sha256(title_key(row["context"][i][0]).encode()).hexdigest() for i in row["gold"])
    quotas, salt = QUOTAS, "kam-native-2wiki-frozen-v1:"
    if holdout is not None:
        for row in json.loads(holdout.read_text())["rows"]:
            held.add(row["id"])
            used_questions.add(tuple(words(row["question"])))
            used_titles.update(u["title_sha256"] for u in row["units"])
        quotas, salt = FIT_QUOTAS, "kam-native-2wiki-in-domain-fit-v1:"
    prior = {"reserved_ids": len(held), "reserved_titles": len(used_titles), "reserved_questions": len(used_questions)}
    groups = defaultdict(list)
    for row in coverage["rows"]["twowiki"]:
        units = row["units"]
        keys = {u["title_sha256"] for u in units}
        q = wording[row["id"]]
        joined = tuple(words(q))
        if (row["id"] in held or not joined or joined in used_questions or keys & used_titles or len(keys) != 2
                or not all(u["indexed"] for u in units) or row["type"] not in quotas):
            continue
        exact = all(u["exact_unit"] for u in units)
        groups[row["type"], exact].append((row, q, joined, keys))
    for group in groups.values():
        group.sort(key=lambda x: stable(salt + x[0]["id"]))
    panels = []
    for kind, n in quotas.items():
        partial = defaultdict(list)
        for row in groups[kind, False]:
            partial[len(row[0]["units"])].append(row)
        count = 0
        for complete in groups[kind, True]:
            _, _, q, titles = complete
            if titles & used_titles or q in used_questions:
                continue
            partial_case = next((row for row in partial[len(complete[0]["units"])]
                                 if row[3].isdisjoint(used_titles) and row[3].isdisjoint(titles)
                                 and row[2] != q and row[2] not in used_questions), None)
            if partial_case is None:
                continue
            for (row, texts, joined, keys), label in ((complete, "exact"), (partial_case, "partial")):
                panels.append({**row, "question": texts, "cohort": label, "match_id": complete[0]["id"]})
                used_titles.update(keys)
                used_questions.add(joined)
            count += 1
            if count == n:
                break
        assert count == n, (kind, count, n)
    assert len(panels) == 2 * sum(quotas.values()) and len({r["id"] for r in panels}) == len(panels)
    panels.sort(key=lambda r: stable("kam-native-2wiki-order-v1:" + r["id"]))
    return {"source_sha256": digest(source), "coverage_sha256": digest(coverage_path), "corpus": coverage["corpus"],
            "reservations": [digest(p) for p in reservations], **prior, "rows": panels} | (
                {"held_out_fixture_sha256": digest(holdout)} if holdout is not None else {})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "coverage", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reserve", type=Path, action="append", required=True)
    p.add_argument("--holdout", type=Path)
    args = p.parse_args()
    result = build(args.source, args.coverage, args.reserve, args.holdout)
    args.output.write_text(json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
