#!/usr/bin/env python3
"""Freeze balanced integer/noninteger pairs from the remaining original MATH test set."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from kam_adapter_preservation import digest
from kam_math_kl_protocol import SOURCES, normalized, write_json

PROTOCOL = "kam-math-balanced-answer-types-v1"
SOURCE_REVISION = "7ecc794703b2877f63226f2477a49b34f9b25163"
SOURCE_SHA256 = "90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076"
PAIR_LIMIT_PER_CELL = 7


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare(source, exclusions, output):
    from grade_kam_full_math500 import check_environment, parse_gold, VERIFY, verify

    check_environment()
    assert digest(source) == SOURCE_SHA256
    for name, sha in SOURCES.items():
        assert digest(exclusions / name) == sha
    inherited = read_jsonl(exclusions / "math500-test.jsonl")
    gsm = read_jsonl(exclusions / "gsm8k-train.jsonl") + read_jsonl(exclusions / "gsm8k-test.jsonl")
    svamp = json.loads((exclusions / "SVAMP.json").read_text())
    blocked = {normalized(row["problem"]) for row in inherited}
    blocked.update(normalized(row["question"]) for row in gsm)
    blocked.update(normalized(row["Body"] + " " + row["Question"]) for row in svamp)
    used_ids = {row["unique_id"] for row in inherited}
    test_rows = [row for row in read_jsonl(source) if row["unique_id"].startswith("test/")]
    by_id = {}
    for row in test_rows:
        item = {field: row[field] for field in ("unique_id", "subject", "level", "problem", "answer")}
        assert by_id.setdefault(item["unique_id"], item) == item
    candidates = list(by_id.values())
    assert len(candidates) == 4500
    assert used_ids.isdisjoint(row["unique_id"] for row in candidates)
    counts = Counter(normalized(row["problem"]) for row in candidates)
    cohorts = defaultdict(lambda: {"integer": [], "noninteger": []})
    dropped = Counter()
    for row in candidates:
        prompt = normalized(row["problem"])
        if prompt in blocked or counts[prompt] > 1:
            dropped["exact_overlap_or_duplicate"] += 1
            continue
        answer = parse_gold(row["answer"])
        if len(answer) != 1 or not verify(answer, answer, **VERIFY):
            dropped["unverified_gold"] += 1
            continue
        kind = "integer" if re.fullmatch(r"-?[0-9]+", row["answer"]) else "noninteger"
        cohorts[row["subject"], row["level"]][kind].append(row)
    cases, pairs, inventory = [], [], []
    for (subject, level), types in sorted(cohorts.items()):
        n = min(PAIR_LIMIT_PER_CELL, *(len(rows) for rows in types.values()))
        inventory.append({"subject": subject, "level": level, "pairs": n,
                          "eligible": {kind: len(rows) for kind, rows in types.items()}})
        ranked = {
            kind: sorted(rows, key=lambda row: hashlib.sha256(
                f"{PROTOCOL}:{kind}:{row['unique_id']}".encode()
            ).digest())[:n]
            for kind, rows in types.items()
        }
        for integer, noninteger in zip(ranked["integer"], ranked["noninteger"], strict=True):
            for row in (integer, noninteger):
                cases.append(row)
            pairs.append({"subject": subject, "level": level,
                          "integer_id": integer["unique_id"], "noninteger_id": noninteger["unique_id"]})
    assert len(cases) == len({row["unique_id"] for row in cases}) == 2 * len(pairs)
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "math-replication.json"
    write_json(fixture, cases)
    write_json(output / "manifest.json", {"protocol": PROTOCOL, "source_revision": SOURCE_REVISION,
                                          "source_sha256": SOURCE_SHA256, "source_test_rows": len(test_rows),
                                          "source_unique_test_ids": len(candidates),
                                          "fixture_sha256": digest(fixture), "n": len(cases), "pairs": pairs,
                                          "pairs_per_cell_max": PAIR_LIMIT_PER_CELL, "inventory": inventory,
                                          "excluded_source_hashes": SOURCES, "dropped": dict(dropped)})


def load_replication(source, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"] == PROTOCOL and manifest["fixture_sha256"] == digest(source)
    rows = json.loads(source.read_text())
    ids = [ident for p in manifest["pairs"] for ident in (p["integer_id"], p["noninteger_id"])]
    assert [row["unique_id"] for row in rows] == ids and len(rows) == manifest["n"]
    for pair, integer, noninteger in zip(manifest["pairs"], rows[::2], rows[1::2], strict=True):
        for item in (integer, noninteger):
            assert (item["subject"], item["level"]) == (pair["subject"], pair["level"])
        assert re.fullmatch(r"-?[0-9]+", integer["answer"])
        assert not re.fullmatch(r"-?[0-9]+", noninteger["answer"])
    return rows


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("exclusions", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    prepare(args.source, args.exclusions, args.output)
