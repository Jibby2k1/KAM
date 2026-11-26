#!/usr/bin/env python3
"""Freeze entity-disjoint TempReason L2 and L3 probes with matched distractors."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from audit_kam_natural_data import digest, stable, words


def question_key(question):
    return " ".join(words(question))


def setup(source, level, limit):
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    raw_entities, picked, used = set(), [], {}
    donors = defaultdict(list)
    for row in rows:
        group = row["id"].split("_")
        assert group[0] == f"L{level}"
        entity, prop = group[1:3]
        raw_entities.add(entity)
        fact_lines = [line.strip() for line in row["fact_context"].splitlines() if line.strip()]
        if len(fact_lines) < 2 or not row["text_answers"]["text"]:
            continue
        donors[prop].append((entity, fact_lines))
        used[row["id"]] = {
            "id": row["id"], "entity": entity, "property": prop,
            "question": row["question"], "answers": row["text_answers"]["text"],
            "facts": fact_lines, "metadata_date": row["date"],
        }
    unique = set()
    for row in sorted(used.values(), key=lambda d: stable("kam-tempreason-v1:" + d["id"])):
        if row["entity"] in unique:
            continue
        noise, other_entities = [], set()
        qtext = question_key(row["question"])
        gold = [f" {question_key(v)} " for v in row["answers"]]
        key = row["id"]
        sorted_donors = sorted(donors[row["property"]], key=lambda r: stable(key + r[0]))
        for entity, fact_lines in sorted_donors:
            if entity == row["entity"] or entity in other_entities:
                continue
            ordered = sorted(fact_lines, key=lambda f: stable(f"{key}:{entity}:{f}"))
            for line in ordered:
                lx = f" {question_key(line)} "
                if line not in row["facts"] and not any(value in lx for value in gold) and qtext not in lx:
                    other_entities.add(entity)
                    noise.append(line)
                    break
            if len(noise) == 8:
                break
        if len(noise) < 8:
            continue
        unique.add(row["entity"])
        picked.append({**row, "noise": noise})
        if len(picked) == limit:
            break
    assert len(picked) >= 160
    return {"level": level, "source_sha256": digest(source), "source_rows": len(rows),
            "source_entities": len(raw_entities), "selected_entities": len(picked),
            "property_counts": dict(Counter(r["property"] for r in picked)), "rows": picked}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--level", type=int, choices=(2, 3), required=True)
    p.add_argument("--limit", type=int, default=384)
    a = p.parse_args()
    data = setup(a.source, a.level, a.limit)
    a.output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print({k: v for k, v in data.items() if k != "rows"})


if __name__ == "__main__":
    main()
