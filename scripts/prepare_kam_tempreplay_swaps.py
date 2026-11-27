#!/usr/bin/env python3
"""Freeze same-multiset duration swaps for hard serial UnSeenTimeQA."""

import argparse
from bisect import bisect_right
from collections import Counter
from hashlib import sha256
from itertools import accumulate, combinations
import json
from pathlib import Path
from random import Random

from audit_kam_unseentimeqa import DAY, INVENTORY, clocks, package_trace, question_key
from audit_kam_unseentimeqa_serial import ANCHOR_END, DURATION, TARGET, serial_events, solve


def digest(data):
    return sha256(data).hexdigest()


def swap(values, i, j):
    result = values.copy()
    result[i], result[j] = result[j], result[i]
    return result


def replace_duration(line, minutes):
    rewritten, count = DURATION.subn(lambda m: m[0].replace(m[1], str(minutes), 1), line)
    assert count == 1
    return rewritten


def construct(row, rng):
    labels, anchor, gold = solve(row, row["question"])
    assert labels == sorted(row["answers"])
    if len(gold) != 1:
        return None
    [original] = gold
    text = row["events"][1:]
    events = serial_events(text)
    if events[-1].end >= DAY or original <= anchor:
        return None
    durations = [e.end - e.start for e in events]
    initial_clause, question = TARGET.split(row["question"], maxsplit=1)
    _, anchor_text = ANCHOR_END.split(initial_clause, maxsplit=1)
    [anchor_clock] = clocks(anchor_text)
    package, target_clock = question_key(question)
    targets_at = (target_clock - anchor_clock + events[anchor].start) % DAY
    [initial] = [ids[1] for line in row["initial_states_description"]
                 if len(ids := INVENTORY.findall(line)) == 2 and ids[0] == package]
    by_event = {i: answer for _, answer, i in package_trace(events, package, initial)}
    assert events[original].start <= targets_at < events[original].end
    options = [(i, j) for i in range(anchor, original) for j in range(original + 1, len(events))
               if durations[i] != durations[j]]
    rng.shuffle(options)
    for i, j in options:
        counterfactual = swap(durations, i, j)
        changed = bisect_right(list(accumulate(counterfactual)), targets_at)
        if changed not in by_event or by_event[changed] == frozenset(labels):
            continue
        delta = abs(durations[i] - durations[j])
        shams = [(a, b) for a, b in combinations(range(max(original, changed) + 1, len(events)), 2)
                 if abs(durations[a] - durations[b]) == delta]
        if shams:
            a, b = rng.choice(shams)
            break
    else:
        return None

    variants = {"original": (text, original, labels, []),
                "sham": (text.copy(), original, labels, [a, b]),
                "intervention": (text.copy(), changed, sorted(by_event[changed]), [i, j])}
    result = {}
    for condition, (lines, expected, answer, slots) in variants.items():
        if slots:
            u, v = slots
            lines[u] = replace_duration(lines[u], durations[v])
            lines[v] = replace_duration(lines[v], durations[u])
        checked = serial_events(lines)
        assert sorted(e.end - e.start for e in checked) == sorted(durations)
        candidate = row | {"events": [row["events"][0], *lines]}
        new_labels, new_anchor, matching = solve(candidate, row["question"])
        assert (new_labels, new_anchor, matching) == (answer, anchor, (expected,))
        result[condition] = {"events": lines, "gold": expected, "labels": answer, "swap": slots}
    plan = [(e.vehicle, e.action, e.locations, e.package) for e in events]
    return {"question": row["question"], "plan": digest(repr(plan).encode()), "anchor": anchor,
            "absolute_delta_minutes": delta, "variants": result}


def build(data, revision):
    sources, output = [], []
    seen = set()
    rng = Random(721136)
    counts = Counter()
    for kind in (1, 2):
        for file in sorted(data.glob(f"*_type_{kind}_split_?.json")):
            raw = file.read_bytes()
            sources.append({"name": file.name, "sha256": digest(raw)})
            for row in json.loads(raw):
                counts[f"type{kind}_source"] += 1
                identity = digest(json.dumps([row["question"], row["events"]]).encode())
                if identity in seen:
                    counts[f"type{kind}_duplicates"] += 1
                    continue
                seen.add(identity)
                record = construct(row, rng)
                if record is not None:
                    counts[f"type{kind}_kept"] += 1
                    output.append({"id": f"{file.name}:{row['id']}", "kind": kind, **record})
    return {"revision": revision, "sources": sources, "counts": dict(counts), "rows": output}


def manifest(fixture, fixture_sha256):
    return {k: fixture[k] for k in ("revision", "sources", "counts")} | {
        "fixture_sha256": fixture_sha256,
        "rows": [{k: case[k] for k in ("id", "kind", "plan", "anchor", "absolute_delta_minutes")} | {
            "events": len(case["variants"]["original"]["events"]),
            "variants": {k: {key: v[key] for key in ("gold", "labels", "swap")}
                         for k, v in case["variants"].items()},
        } for case in fixture["rows"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = build(args.source, args.revision)
    payload = (json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    args.output.write_bytes(payload)
    if args.manifest:
        projected = manifest(result, digest(payload))
        args.manifest.write_text(json.dumps(projected, separators=(",", ":")) + "\n")
    print(result["counts"])


if __name__ == "__main__":
    main()
