#!/usr/bin/env python3
"""Inventory a pinned independent math source without evaluating a model."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

from math_verify.errors import TimeoutException

from grade_kam_full_math500 import VERIFY, check_environment, parse_gold, verify
from kam_adapter_preservation import digest
from kam_math_answer_type_replication import SOURCE_SHA256 as PRM_TRAIN_SHA256, read_jsonl
from kam_math_kl_protocol import SOURCES, normalized, write_json

REVISION = "5aac08bc8ebeef5511e244c094194321209550af"
SOURCE_SHA256 = "7c87be8ee41ac7c7a597ef5a5500e84bd2b639a85a06db3da7f69bf9a32ef168"
SCALAR = re.compile(r"[+-]?\d+(?:\.\d+)?")


def gold_status(answer):
    text = answer.strip()
    if not text:
        return "empty"
    try:
        parsed = parse_gold(text)
        if len(parsed) == 1 and verify(parsed, parsed, **VERIFY):
            return "plain_scalar" if SCALAR.fullmatch(text.strip("$")) else "other_self_verifying"
    except TimeoutException:
        return "unverified"
    return "unverified"


def difficulty_band(level):
    if level <= 2:
        return "at_most_2"
    if level <= 4:
        return "above_2_to_4"
    if level <= 6:
        return "above_4_to_6"
    return "above_6"


def prior_prompts(prior, prm_source):
    assert digest(prm_source) == PRM_TRAIN_SHA256
    assert all(digest(prior / name) == sha for name, sha in SOURCES.items())
    prm = read_jsonl(prm_source)
    return {
        "prm_original_math_train": {normalized(row["problem"]) for row in prm if row["unique_id"].startswith("train/")},
        "prm_original_math_test": {normalized(row["problem"]) for row in prm if row["unique_id"].startswith("test/")},
        "math500": {normalized(row["problem"]) for row in read_jsonl(prior / "math500-test.jsonl")},
        "gsm8k": {normalized(row["question"]) for name in ("gsm8k-train.jsonl", "gsm8k-test.jsonl")
                  for row in read_jsonl(prior / name)},
        "svamp": {normalized(row["Body"] + " " + row["Question"])
                  for row in json.loads((prior / "SVAMP.json").read_text())},
    }


def inventory(source, prior, prm_source, workers):
    check_environment()
    assert digest(source) == SOURCE_SHA256
    rows = read_jsonl(source)
    assert len(rows) == 4428 and all(row["problem"].strip() for row in rows)
    groups = prior_prompts(prior, prm_source)
    all_prior = set().union(*groups.values())
    keys = [normalized(row["problem"]) for row in rows]
    repeats = Counter(keys)
    overlaps = {group: sum(key in values for key in keys) for group, values in groups.items()}
    gold_totals = Counter()
    difficulty = defaultdict(Counter)
    contest = defaultdict(Counter)
    clean = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        statuses = pool.map(gold_status, (r["answer"] for r in rows), chunksize=24)
        for row, key, status in zip(rows, keys, statuses, strict=True):
            gold_totals[status] += 1
            if repeats[key] > 1 or key in all_prior:
                continue
            clean[status] += 1
            difficulty[difficulty_band(row["difficulty"])][status] += 1
            contest[row["source"]][status] += 1
    return {
        "source_revision": REVISION, "source_sha256": SOURCE_SHA256, "n": len(rows),
        "prior_sha256": {**SOURCES, "prm_source": PRM_TRAIN_SHA256},
        "normalized_exact_overlap_rows": overlaps, "internal_duplicate_rows": sum(n for n in repeats.values() if n > 1),
        "gold_status": gold_totals, "disjoint_unique_gold_status": clean,
        "disjoint_unique_difficulty": difficulty, "disjoint_unique_contest": contest,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("prior_folder", type=Path)
    cli.add_argument("prm_source", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--workers", type=int, default=4)
    args = cli.parse_args()
    write_json(args.output, inventory(args.source, args.prior_folder, args.prm_source, args.workers))
