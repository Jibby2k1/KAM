#!/usr/bin/env python3
"""Recover event indices from duration-only UnSeenTimeQA serial schedules."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import re

from audit_kam_unseentimeqa import DAY, Event, INVENTORY, clocks, package_trace, question_key, translate

DURATION = re.compile(r"\b(\d+)\s+minutes?\b", re.I)
LOAD = re.compile(r"\b(un)?load(?:ed|ing)\b", re.I)
TARGET = re.compile(r"\bwhere\s+is\b", re.I)
ANCHOR_END = re.compile(r"\s+starts\s+at\s+", re.I)
CHANGE = re.compile(r"\band\s+(.+?)\s+is\s+(delayed|expedited)\s+by\s+(\d+)\s+minutes?\b", re.I)


def arguments(text):
    identifiers = INVENTORY.findall(text)
    vehicles = [i for i in identifiers if i[0] in "at"]
    locations = tuple(i for i in identifiers if i[0] == "l")
    packages = [i for i in identifiers if i[0] == "p"]
    load = LOAD.search(text)
    if len(vehicles) == 1 and len(locations) == 1 and len(packages) == 1 and load:
        return vehicles[0], "unload" if load[1] else "load", locations, packages[0]
    if len(vehicles) == 1 and len(locations) == 2 and not packages and not load:
        return vehicles[0], "travel", locations, None
    raise ValueError("event_arguments")


def serial_events(texts, change=None, delta=0):
    out = []
    matches = 0
    for text in texts:
        durations = DURATION.findall(text)
        if len(durations) != 1 or int(durations[0]) < 1 or clocks(text):
            raise ValueError("event_duration")
        start = out[-1].end if out else 0
        signature = arguments(text)
        duration = int(durations[0])
        if signature == change:
            duration += delta
            matches += 1
        if duration < 0:
            raise ValueError("negative_changed_duration")
        out.append(Event(start, start + duration, *signature))
    if not out:
        raise ValueError("empty_events")
    if change is not None and matches != 1:
        raise ValueError("ambiguous_changed_event")
    return out


def solve(row, question, *, apply_change=True):
    parts = TARGET.split(question, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("query_shape")
    anchor_clause, target_clause = parts
    parts = ANCHOR_END.split(anchor_clause, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("anchor_shape")
    anchor_text, anchor_clock_text = parts
    timestamps = clocks(anchor_clock_text)
    if len(timestamps) != 1:
        raise ValueError("anchor_clock")
    signature = arguments(anchor_text)
    change = CHANGE.search(anchor_clock_text)
    package, query_clock = question_key(target_clause)
    states = [i[1] for line in row["initial_states_description"]
              if len(i := INVENTORY.findall(line)) == 2 and i[0] == package]
    if len(states) != 1:
        raise ValueError("initial_state")
    if change and apply_change:
        text, direction, minutes = change.groups()
        delta = int(minutes) * (1 if direction.lower() == "delayed" else -1)
        events = serial_events(row["events"][1:], arguments(text), delta)
    else:
        events = serial_events(row["events"][1:])
    anchors = [(i, event.start) for i, event in enumerate(events)
               if (event.vehicle, event.action, event.locations, event.package) == signature]
    if len(anchors) != 1:
        raise ValueError("ambiguous_anchor")
    anchor, source_minute = anchors[0]
    target = (query_clock - timestamps[0] + source_minute) % DAY
    eligible = [(index, labels) for e, labels, index in package_trace(events, package, states[0])
                if (target - e.start) % DAY < e.end - e.start]
    answer_sets = {labels for _, labels in eligible}
    if len(answer_sets) != 1:
        raise ValueError("ambiguous_or_missing_target")
    return sorted(answer_sets.pop()), anchor, tuple(i for i, _ in eligible)


def audit(data, revision):
    rng = Random(79551)
    summary, scores = {}, []
    for kind in (1, 2, 3):
        counts = Counter()
        unmatched = Counter()
        worlds, schedules = set(), set()
        for path in sorted(data.glob(f"*_type_{kind}_split_?.json")):
            raw = path.read_bytes()
            rows = json.loads(raw)
            scores.append({"file": path.name, "sha256": sha256(raw).hexdigest(), "n": len(rows)})
            assert len(rows) == len({r["id"] for r in rows}) == 300
            for row in rows:
                assert (CHANGE.search(row["question"]) is not None) == (kind == 3)
                worlds.add(tuple(sorted(map(tuple, map(INVENTORY.findall, row["initial_states_description"])))))
                schedules.add(tuple(arguments(line) for line in row["events"][1:]))
                counts["n"] += 1
                delta = rng.randrange(1, DAY)
                try:
                    result = solve(row, row["question"])
                except ValueError as error:
                    unmatched[str(error)] += 1
                    continue
                answer, anchor, events = result
                counts["solved"] += 1
                counts["author_exact"] += answer == sorted(row["answers"])
                counts["depth_matches"] += any(i - anchor == row["depth"] for i in events)
                counts["same_after_clock_shift"] += solve(row, translate(row["question"], delta)) == result
                if kind == 3:
                    try:
                        unchanged, _, _ = solve(row, row["question"], apply_change=False)
                    except ValueError:
                        counts["unchanged_abstains"] += 1
                    else:
                        counts["unchanged_author_exact"] += unchanged == sorted(row["answers"])
        assert counts["n"] == 900
        summary[str(kind)] = {"counts": dict(counts), "initial_worlds": len(worlds),
                              "action_sequences": len(schedules), "unmatched": dict(sorted(unmatched.items()))}
    return {"revision": revision, "files": scores, "results": summary}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(audit(a.source, a.revision), indent=2) + "\n")


if __name__ == "__main__":
    main()
