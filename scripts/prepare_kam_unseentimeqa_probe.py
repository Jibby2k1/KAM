#!/usr/bin/env python3
"""Freeze source- and time-matched event decoys from separate fictional schedules."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from audit_kam_unseentimeqa import DAY, parse_events, question_key, solve, translate


def build(root):
    records = defaultdict(list)
    registry = defaultdict(list)
    for difficulty in ("easy", "medium"):
        for kind in (1, 2):
            for source in sorted((root / difficulty).glob(f"*_type_{kind}_split_?.json")):
                data = json.loads(source.read_text())
                for row in data:
                    signature = "\n".join(sorted(row["initial_states_description"]))
                    plan = sha256(signature.encode()).hexdigest()
                    labels, indices = solve(row)
                    assert labels == sorted(row["answers"])
                    parsed = parse_events(row["events"][1:])
                    for event, text in zip(parsed, row["events"][1:], strict=True):
                        if event.package is not None:
                            registry[event.package].append((plan, event, text))
                    package, clock = question_key(row["question"])
                    if len(indices) == 1:
                        records[difficulty, kind].append((plan, source.name, row, parsed, indices[0], package, clock))
    rng = Random(7483159)
    output = []
    for (difficulty, kind), cases in sorted(records.items()):
        seen = set()
        counts = defaultdict(int)
        ordered = list(cases)
        rng.shuffle(ordered)
        for plan, source, row, parsed, gold, package, clock in ordered:
            qkey = plan, row["question"], tuple(row["events"])
            if counts[plan] == 18 or qkey in seen:
                continue
            # Keep the corruption and sham physically identical except for the
            # donor event clock, with the same duration in both conditions.
            candidates = registry[package]
            for ix in rng.sample(range(len(candidates)), min(100, len(candidates))):
                other_plan, event, donor = candidates[ix]
                duration = event.end - event.start
                if other_plan != plan and duration < 180:
                    break
            else:
                continue
            shift = clock - duration // 2 - event.start
            colliding, distant = translate(donor, shift), translate(donor, shift + 240)
            observed = [(clock - e.start) % DAY < e.end - e.start for e in parsed]
            if sum(observed) != 1 or not observed[gold]:
                continue
            hour, minute = divmod(clock, 60)
            explicit = f"Where is the package {package} at {hour % 12 or 12:02d}:{minute:02d} {'AM' if hour < 12 else 'PM'}?"
            output.append({"id": f"{source}:{row['id']}", "scope": plan, "other_scope": other_plan,
                           "difficulty": difficulty, "kind": kind, "question": row["question"],
                           "resolved_question": explicit, "events": row["events"][1:], "gold_event": gold,
                           "gold_in_time": observed, "sham": distant, "collision": colliding})
            seen.add(qkey)
            counts[plan] += 1
            if sum(counts.values()) == 150:
                break
        assert sum(counts.values()) == 150, (difficulty, kind, len(counts))
    assert len(output) == 600 and len({r["id"] for r in output}) == 600
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = build(a.root)
    a.output.write_text(json.dumps({"rows": rows}, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
