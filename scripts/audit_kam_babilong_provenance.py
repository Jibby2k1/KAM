#!/usr/bin/env python3
"""Replay matched fact-shaped book noise with and without source admission."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from random import Random

from analyze_kam_native_musique import compare
from audit_kam_babilong_replay import ACTOR, ITEM, LOCATION, TASKS, find_events, replay
from audit_kam_natural_data import digest

REPETITIONS = 8


def draws(task, id_):
    rng = Random(f"kam-babilong-source-v1:{id_}")
    possible_actors, items, rooms = [s.split("|") for s in (ACTOR, ITEM, LOCATION)]
    for _ in range(REPETITIONS):
        actor, item = rng.choice(possible_actors), rng.choice(items)
        first, second = rng.sample(rooms, 2)
        take = f"{actor} picked up the {item}."
        move = f"{actor} went to the {first}."
        if task == "qa2":
            actions = move, take
        elif task == "qa3":
            actions = take, move, f"{actor} went to the {second}."
        else:
            actions = (take,)
        yield "A fictional example in a different account follows. " + " ".join(actions)


def case_outcome(case):
    trace = find_events(case["text"])
    task, question = case["task"], case["question"]
    gold = tuple(case["gold"]) if isinstance(case["gold"], list) else case["gold"]
    native, original_witness = replay(task, question, trace)
    assert native == gold
    trace_stamps, rows = [], []
    for note in draws(task, case["id"]):
        add = find_events(note)
        assert len(add) == {"qa2": 2, "qa3": 3, "qa7": 1, "qa8": 1}[task]
        assert len({(a,b) for a,b,_ in add}) == len(add)
        result = {}
        for mode in ("prefix", "suffix"):
            source, offset = ((note + "\n\n" + case["text"], len(note) + 2) if mode == "prefix"
                              else (case["text"] + "\n\n" + note, 0))
            flat = find_events(source)
            keep = [v for v in flat if offset <= v[0] and v[1] <= offset + len(case["text"])]
            assert len(flat) == len(keep) + len(add)
            filtered, witnesses = replay(task, question, keep)
            assert filtered == gold and witnesses == original_witness
            raw, _ = replay(task, question, flat)
            result[mode] = {"unguarded_exact": raw == gold, "authoritative_exact": filtered == gold}
        trace_stamps.append([fields for _, _, fields in add])
        rows.append(result)
    return {"id": case["id"], "task": task, "authoritative_event_count": len(trace),
            "draws": rows, "decoy_actions": trace_stamps}


def summarize(rows):
    result = {}
    for task in TASKS:
        subset = [q for q in rows if q["task"] == task]
        raw = [r for q in subset for r in q["draws"]]
        groups = [list(range(i * REPETITIONS, (i+1) * REPETITIONS)) for i in range(len(subset))]
        totals = {}
        for mode in ("prefix", "suffix"):
            totals[mode] = Counter()
            for item in raw:
                totals[mode].update(item[mode])
        x, y = [[r[k]["unguarded_exact"] for r in raw] for k in ("prefix", "suffix")]
        result[task] = {"episodes": len(subset), "trials": len(raw), "counts": totals,
                        "suffix_minus_prefix": compare(y, x, groups), "clean": len(raw)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("fixture", "output", "replay"):
        p.add_argument("--" + opt, type=Path, required=True)
    a = p.parse_args()
    with gzip.open(a.fixture, "rt") as source:
        records = json.load(source)
    cases = [q for q in records["rows"] if q["length"] == "64k"]
    assert len(cases) == len({q["id"] for q in cases}) == 386
    audit = [case_outcome(q) for q in cases]
    result = {"original_fixture_sha256": digest(a.fixture), "draws_per_episode": REPETITIONS,
              "source_revision": records["revision"], "tasks": summarize(audit)}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    a.replay.write_bytes(gzip.compress(json.dumps(audit, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
