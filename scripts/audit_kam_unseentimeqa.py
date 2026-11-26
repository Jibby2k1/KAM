#!/usr/bin/env python3
"""Audit whether serial UnSeenTimeQA can provide independent executable labels."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import re


DAY = 24 * 60
CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\s*(AM|PM)\b", re.I)
INVENTORY = re.compile(r"\b(?:p\d+|[at]\d+|l\d+_\d+)\b")
QUESTION = re.compile(r"\b(?:package|product)\s+(p\d+)\b", re.I)
RELATIVE = re.compile(r"\b(\d+)\s+hours?\s+(before|after)\b", re.I)


@dataclass(frozen=True)
class Event:
    start: int
    end: int
    vehicle: str
    action: str
    locations: tuple[str, ...]
    package: str | None


def minute(match):
    hours, minutes, phase = match.groups()
    if not (1 <= int(hours) <= 12 and 0 <= int(minutes) < 60):
        raise ValueError("invalid12hour")
    return (int(hours) % 12) * 60 + int(minutes) + 720 * (phase.upper() == "PM")


def clocks(text):
    return [minute(match) for match in CLOCK.finditer(text)]


def translate(text, shift):
    def offset(match):
        hour, minutes = divmod((minute(match) + shift) % DAY, 60)
        return f"{hour % 12 or 12:02d}:{minutes:02d} {'AM' if hour < 12 else 'PM'}"
    return CLOCK.sub(offset, text)


def parse_events(texts):
    events = []
    for text in texts:
        times = clocks(text)
        ids = INVENTORY.findall(text)
        vehicles = [i for i in ids if i[0] in "at"]
        locations = tuple(i for i in ids if i[0] == "l")
        packages = [i for i in ids if i[0] == "p"]
        if len(vehicles) != 1 or len(times) not in (1, 2):
            raise ValueError("event_time_or_vehicle")
        if len(times) == 2:
            duration = (times[1] - times[0]) % DAY
        else:
            authority = re.findall(r"\b(\d+)\s+minutes?\b", text, re.I)
            if len(authority) != 1:
                raise ValueError("duration")
            duration = int(authority[0])
        if duration <= 0:
            raise ValueError("empty_interval")
        start = times[0]
        previous = events[-1].end if events else start
        if start < previous:
            start += ((previous - start + DAY - 1) // DAY) * DAY
        if start - previous >= 60:
            raise ValueError("unclear_serial_order")
        if len(packages) == 1 and len(locations) == 1 and re.search(r"\b(?:un)?load(?:ed|ing)\b", text, re.I):
            action = "unload" if re.search(r"\bunload(?:ed|ing)\b", text, re.I) else "load"
            package = packages[0]
        elif not packages and len(locations) == 2:
            action, package = "travel", None
        else:
            raise ValueError("event_arguments")
        events.append(Event(start, start + duration, vehicles[0], action, locations, package))
    if not events:
        raise ValueError("empty")
    return events


def package_trace(events, package, initial):
    hits = []
    state = initial
    for i, event in enumerate(events):
        if event.package == package:
            location = event.locations[0]
            if event.action == "load":
                if state != location:
                    raise ValueError("inconsistent_load")
                state = event.vehicle
            elif event.action == "unload":
                if state != event.vehicle:
                    raise ValueError("inconsistent_unload")
                state = location
            hits.append((event, frozenset((location, event.vehicle)), i))
        elif event.vehicle == state:
            # The author labels use all cargo present for an event. During
            # loading or unloading, the label adds the stationary location.
            labels = (state,) if event.action == "travel" else (state, event.locations[0])
            hits.append((event, frozenset(labels), i))
    return hits


def question_key(question):
    times = clocks(question)
    package = QUESTION.findall(question)
    if len(times) != 1 or len(package) != 1:
        raise ValueError("question_shape")
    package = package[0]
    offset = RELATIVE.search(question)
    target = times[0]
    if offset:
        delta = int(offset[1]) * 60
        target += delta if offset[2].lower() == "after" else -delta
    return package, target % DAY


def solve(row):
    package, target = question_key(row["question"])
    states = [ids[1] for s in row["initial_states_description"]
              if len(ids := INVENTORY.findall(s)) == 2 and ids[0] == package]
    if len(states) != 1:
        raise ValueError("initial_state")
    events = parse_events(row["events"][1:])
    trace = package_trace(events, package, states[0])
    # Select occurrences only where the queried package is participating, as
    # the author generator samples event-interior questions. Day is unspecified.
    candidates = [(labels, index) for e, labels, index in trace if (target - e.start) % DAY < e.end - e.start]
    distinct = set(labels for labels, _ in candidates)
    if not distinct:
        raise ValueError("no_participating_event")
    if len(distinct) > 1:
        raise ValueError("ambiguous_day")
    return sorted(distinct.pop()), tuple(index for _, index in candidates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    cfg = p.parse_args()
    detail, sources = {}, []
    rng = Random(59233)
    for difficulty in ("easy", "medium"):
        for kind in (1, 2):
            counts, reason, depths = Counter(), Counter(), Counter()
            for path in sorted((cfg.root / difficulty).glob(f"*_type_{kind}_split_?.json")):
                raw = path.read_bytes()
                sources.append({"file": str(path.relative_to(cfg.root)), "bytes": len(raw),
                                "sha256": sha256(raw).hexdigest()})
                rows = json.loads(raw)
                assert len(rows) == 300
                for row in rows:
                    shift = rng.randrange(1, DAY)
                    shifted = {**row, "question": translate(row["question"], shift),
                               "events": [translate(x, shift) for x in row["events"]]}
                    counts["n"] += 1
                    depths[row["depth"]] += 1
                    try:
                        labels, indices = solve(row)
                    except ValueError as failure:
                        reason[str(failure)] += 1
                        continue
                    counts["solved"] += 1
                    counts["author_exact"] += labels == sorted(row["answers"])
                    counts["author_intersection"] += bool(set(labels) & set(row["answers"]))
                    counts["same_after_clock_shift"] += solve(shifted) == (labels, indices)
            assert counts["n"] == 900
            detail[f"{difficulty}:{kind}"] = dict(counts) | {"non_solved": dict(reason), "depth_counts": dict(depths)}
    cfg.output.write_text(json.dumps({"revision": cfg.revision, "sources": sources, "conditions": detail}, indent=2) + "\n")


if __name__ == "__main__":
    main()
