#!/usr/bin/env python3
"""Resolve typed BABILong events through object and actor postings."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from audit_kam_babilong_replay import QUESTIONS, find_events, replay
from audit_kam_natural_data import digest
from run_kam_babilong_retrieval import BUDGETS, chunks, evaluate


class EventIndex:
    """Index recognized events once, preserving their source order."""

    def __init__(self, events):
        self.events = events
        self.moves = defaultdict(list)
        self.objects = defaultdict(list)
        for i, (_, _, event) in enumerate(events):
            if event["place"]:
                self.moves[event["actor"]].append(i)
            else:
                item = event["take"] or event["drop"] or event["give"]
                self.objects[item].append(i)

    def path(self, item):
        transitions = self.objects.get(item, ())
        sequence = []

        def append(move, event):
            place = self.events[move][2]["place"]
            edge = (place, {event, move})
            if sequence and sequence[-1][0] == place:
                sequence[-1] = edge
            else:
                sequence.append(edge)

        for j, change in enumerate(transitions):
            event = self.events[change][2]
            actor = event["actor"]
            prior = self.moves.get(actor, ())
            position = bisect_left(prior, change)
            if position:
                append(prior[position-1], change)
            if not event["drop"]:
                owner = event["receiver"] or actor
                moves = self.moves.get(owner, ())
                stop = transitions[j+1] if j+1 < len(transitions) else len(self.events)
                for move in moves[bisect_right(moves, change):bisect_left(moves, stop)]:
                    append(move, change)
        return sequence

    def query(self, task, question):
        fields = QUESTIONS[task].fullmatch(question.strip())
        assert fields
        if task in ("qa2", "qa3"):
            route = self.path(fields["object"])
            if task == "qa2":
                return route[-1] if route else (None, set())
            for (prior, first), (new, second) in reversed(list(zip(route, route[1:]))):
                if new == fields["destination"]:
                    return prior, first | second
            return None, set()
        last_updates = {item: indices[-1] for item, indices in self.objects.items()}
        owner = fields["owner"]
        carried = []
        for item, i in sorted(last_updates.items()):
            event = self.events[i][2]
            if not event["drop"] and (event["receiver"] or event["actor"]) == owner:
                carried.append(item)
        return (len(carried) if task == "qa7" else tuple(carried)), set(last_updates.values())


def select(windows, targets, budget):
    cover = [targets & {(lo, hi) for lo, hi, _ in found} for found in windows]
    picked, remain = [], set(targets)
    while remain and len(picked) < budget:
        top = max(range(len(cover)), key=lambda i: (len(cover[i] & remain), -i))
        assert cover[top] & remain
        remain -= cover[top]
        picked.append(top)
    return picked


def measure(case):
    started = perf_counter()
    events = find_events(case["text"])
    index = EventIndex(events)
    prepared = perf_counter()
    prediction, proof = index.query(case["task"], case["question"])
    ready = perf_counter()
    independent, witness = replay(case["task"], case["question"], events)
    gold = tuple(case["gold"]) if isinstance(case["gold"], list) else case["gold"]
    assert independent == gold and len(events) == case["event_count"]
    assert witness == {i for i, (lo, hi, _) in enumerate(events) if [lo, hi] in case["evidence"]}
    found = {(events[i][0], events[i][1]) for i in proof}
    _, windows = chunks(case["text"])
    result = {}
    for budget in BUDGETS:
        retrieved = select(windows, found, budget)
        report = evaluate(case, retrieved, windows)
        report["selected_windows"] = len(retrieved)
        result[str(budget)] = report
    return {"id": case["id"], "task": case["task"], "length": case["length"], "index_exact": prediction == gold,
            "index_witness_superset": witness <= proof, "read_budgets": result, "proof_events": len(proof),
            "full_events": len(events), "index_ms": 1000*(prepared-started), "lookup_ms": 1000*(ready-prepared)}


def summarize(rows):
    result = {}
    for task in sorted({row["task"] for row in rows}):
        arms = {}
        for length in ("16k", "64k"):
            tests = [r for r in rows if r["task"] == task and r["length"] == length]
            assert len(tests) in (86, 100)
            counts = {k: sum(q[k] for q in tests) for k in ("index_exact", "index_witness_superset")}
            budgets = {}
            for d in BUDGETS:
                notes = [q["read_budgets"][str(d)] for q in tests]
                budgets[str(d)] = {k: sum(q[k] for q in notes) for k in ("complete_witness", "exact", "joint")}
                budgets[str(d)]["selected_window_counts"] = dict(Counter(q["selected_windows"] for q in notes))
            arms[length] = {"n": len(tests), "archive_counts": counts, "read_budgets": budgets,
                            "median_index_ms": median(r["index_ms"] for r in tests),
                            "median_lookup_ms": median(r["lookup_ms"] for r in tests)}
        result[task] = arms
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("fixture", "output", "summary"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    with gzip.open(a.fixture, "rt") as stream:
        cases = json.load(stream)
    rows = [measure(q) for q in cases["rows"] if q["length"] in ("16k", "64k")]
    assert len(rows) == 772
    info = {"fixture_sha256": digest(a.fixture), "source_revision": cases["revision"]}
    a.output.write_bytes(gzip.compress(json.dumps(info | {"rows": rows}, separators=(",", ":")).encode(), mtime=0))
    a.summary.write_text(json.dumps(info | {"tasks": summarize(rows)}, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
