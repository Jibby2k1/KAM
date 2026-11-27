#!/usr/bin/env python3
"""Reconstruct concurrent schedules from the UnSeenTimeQA duration-only event text."""

import argparse
import ast
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import re

from audit_kam_unseentimeqa import DAY, INVENTORY, clocks, package_trace, question_key, translate
from audit_kam_unseentimeqa_serial import ANCHOR_END, CHANGE, TARGET, arguments, serial_events, solve as serial_solve

PLAN_ACTION = re.compile(r"^occurs\(action_(\w+)\(([^)]+)\),(\d+)\)\.$", re.M)


def signature(action):
    return action.vehicle, action.action, action.locations, action.package


def graph_reference(root):
    code = root / "scripts/hard_parallel/parallel_instances.py"
    raw = code.read_bytes()
    source = [{"name": str(code.relative_to(root)), "sha256": sha256(raw).hexdigest()}]
    modules = {int(f.name.rsplit("_", 1)[-1]): f for f in ast.parse(raw).body
               if isinstance(f, ast.FunctionDef) and f.name.startswith("populate_instance_")}
    graphs = {}
    for instance, function in modules.items():
        path = root / f"domains/logistics_parallel/instances/Instance_{instance}/plan.lp"
        data = path.read_bytes()
        source.append({"name": str(path.relative_to(root)), "sha256": sha256(data).hexdigest()})
        plan = []
        records = PLAN_ACTION.findall(data.decode())
        assert {int(i) for _, _, i in records} == set(range(1, len(records) + 1))
        for operation, raw_ids, _ in records:
            a, b, c, *_ = raw_ids.split(",")
            group = operation.split("_")[0]
            plan.append((a, "travel", (b, c), None) if group in ("fly", "drive") else (b, group, (c,), a))
        assert plan and tuple(plan) not in graphs
        assert isinstance(function.body[-1], ast.Return)
        graphs[tuple(plan)] = function.body[:-1]
    assert len(graphs) == 10
    return graphs, source


def start_index(node):
    assert isinstance(node, ast.Subscript) and node.slice.value == "current_time"
    assert isinstance(node.value, ast.Subscript) and node.value.value.id == "graph"
    return int(node.value.slice.value)


def graph_schedule(actions, graphs):
    statements = graphs[tuple(map(signature, actions))]
    starts = [0] * len(actions)
    durations = [e.end - e.start for e in actions]

    def value(node):
        if isinstance(node, ast.Subscript):
            return starts[start_index(node)]
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        if node.func.id == "total_time":
            g, idx = node.args
            assert g.id == "graph" and isinstance(idx, ast.Constant)
            i = int(idx.value)
            return starts[i] + durations[i]
        assert node.func.id == "max"
        return max(map(value, node.args))

    for instruction in statements:
        assert isinstance(instruction, ast.Assign) and len(instruction.targets) == 1
        starts[start_index(instruction.targets[0])] = value(instruction.value)
    return [replace(e, start=start, end=start + duration)
            for e, start, duration in zip(actions, starts, durations, strict=True)]


def schedule(actions):
    groups, predecessors = [], []
    last_vehicle, last_package = {}, {}
    for event_id, action in enumerate(actions):
        previous = last_vehicle.get(action.vehicle)
        group_id = len(groups)
        if previous is not None and action.action != "travel":
            first = actions[groups[previous][0]]
            if first.action == action.action and first.locations == action.locations:
                group_id = previous
        if group_id == len(groups):
            groups.append([])
            predecessors.append(set(groups[previous]) if previous is not None else set())
        groups[group_id].append(event_id)
        last_vehicle[action.vehicle] = group_id
        if action.package is not None:
            if action.package in last_package:
                predecessors[group_id].add(last_package[action.package])
            last_package[action.package] = event_id

    timed = {}
    pending = set(range(len(groups)))
    while pending:
        ready = [g for g in sorted(pending) if predecessors[g] <= timed.keys()]
        if not ready:
            raise ValueError("cyclic_event_dependencies")
        for g in ready:
            start = max((timed[e].end for e in predecessors[g]), default=0)
            for event_id in groups[g]:
                e = actions[event_id]
                timed[event_id] = replace(e, start=start, end=start + e.end - e.start)
        pending.difference_update(ready)
    return [timed[i] for i in range(len(actions))]


def location(events, package, initial, target):
    active = [e for e in events if e.start <= target < e.end]
    direct = [e for e in active if e.package == package]
    if direct:
        if len(direct) > 1:
            raise ValueError("overlapping_package_events")
        [event] = direct
        return frozenset((event.vehicle, *event.locations))
    previous = max((e for e in events if e.package == package and e.end <= target),
                   key=lambda e: e.end, default=None)
    holder = initial if previous is None else (previous.vehicle if previous.action == "load" else previous.locations[0])
    labels = {holder}
    for event in active:
        if event.vehicle == holder and event.action != "travel":
            labels.update(event.locations)
    return frozenset(labels)


def solve(row, question, *, graph=None):
    anchor_clause, target_clause = TARGET.split(question, maxsplit=1)
    anchor_text, anchor_clock_text = ANCHOR_END.split(anchor_clause, maxsplit=1)
    [anchor_clock] = clocks(anchor_clock_text)
    anchor = arguments(anchor_text)
    package, query = question_key(target_clause)
    states = [ids[1] for line in row["initial_states_description"]
              if len(ids := INVENTORY.findall(line)) == 2 and ids[0] == package]
    if len(states) != 1:
        raise ValueError("initial_state")
    change = CHANGE.search(anchor_clock_text)
    if change:
        text, direction, minutes = change.groups()
        delta = int(minutes) * (1 if direction.lower() == "delayed" else -1)
        actions = serial_events(row["events"][1:], arguments(text), delta)
    else:
        actions = serial_events(row["events"][1:])
    events = schedule(actions) if graph is None else graph_schedule(actions, graph)
    anchors = [(i, e) for i, e in enumerate(events) if signature(e) == anchor]
    if len(anchors) != 1:
        raise ValueError("ambiguous_anchor")
    anchor_id, origin = anchors[0]
    target = (query - anchor_clock) % DAY + origin.start
    if target > max(e.end for e in events):
        raise ValueError("target_beyond_schedule")
    answering = location(events, package, states[0], target)
    labels_by_event = {i: label for event, label, i in package_trace(events, package, states[0])
                       if event.start <= target < event.end}
    return sorted(answering), anchor_id, target, events, labels_by_event


def audit(data, revision, split, graph_root):
    rng = Random(180179)
    results, sources = {}, []
    graphs, graph_sources = graph_reference(graph_root) if graph_root is not None else (None, [])
    for kind in (1, 2, 3):
        counts, unmatched, per_plan = Counter(), Counter(), {}
        for path in sorted(data.glob(f"*_type_{kind}_split_{split or '?'}.json")):
            raw = path.read_bytes()
            sources.append({"name": path.name, "sha256": sha256(raw).hexdigest()})
            rows = json.loads(raw)
            assert len(rows) == len({r["id"] for r in rows}) == 300
            for row in rows:
                counts["n"] += 1
                q = row["question"]
                target = sorted(row["answers"])
                plan = tuple(arguments(text) for text in row["events"][1:])
                stamp = sha256(repr(plan).encode()).hexdigest()
                plan_counts = per_plan.setdefault(stamp, Counter())
                plan_counts["n"] += 1
                try:
                    serial, _, _ = serial_solve(row, q)
                except ValueError:
                    counts["serial_abstains"] += 1
                else:
                    counts["serial_exact"] += serial == target
                try:
                    result = solve(row, q)
                except ValueError as error:
                    unmatched[str(error)] += 1
                    continue
                counts["solved"] += 1
                counts["author_exact"] += result[0] == target
                plan_counts["author_exact"] += result[0] == target
                event_options = set(result[4].values())
                counts["inferred_unique_event_answer_exact"] += event_options == {frozenset(target)}
                counts["inferred_no_event_answer"] += not event_options
                counts["inferred_multiple_event_answers"] += len(event_options) > 1
                counts["same_after_shift"] += solve(row, translate(q, rng.randrange(1, DAY))) == result
                if graphs is not None:
                    observed, anchor, target_in_graph, events, labels_by_event = solve(row, q, graph=graphs)
                    authored_index = anchor + row["depth"]
                    authored_event = events[authored_index]
                    alternatives = set(labels_by_event.values())
                    counts["reference_graph_exact"] += observed == target
                    counts["reference_graph_contains_authored_event"] += authored_event.start <= target_in_graph < authored_event.end
                    counts["reference_graph_authored_event_labels_match"] += labels_by_event.get(authored_index) == frozenset(target)
                    counts["reference_graph_answers_with_one_event_label_set"] += alternatives == {frozenset(target)}
                    counts["reference_graph_answers_with_multiple_event_label_sets"] += len(alternatives) > 1
                    counts["schedules_identical"] += result[3] == events
        assert counts["n"] == (300 if split else 900)
        results[str(kind)] = {"counts": dict(counts), "unmatched": dict(sorted(unmatched.items())),
                              "plans": dict(sorted(per_plan.items()))}
    return {"revision": revision, "sources": sources, "graph_sources": graph_sources, "results": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--revision", required=True)
    p.add_argument("--split", choices=(1, 2, 3), type=int)
    p.add_argument("--graph-root", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(audit(a.source, a.revision, a.split, a.graph_root), indent=2) + "\n")


if __name__ == "__main__":
    main()
