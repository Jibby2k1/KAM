#!/usr/bin/env python3
"""Compare fixed-budget hard routes on matched BABILong filler lengths."""

import argparse
import gzip
import json
from pathlib import Path
import re
from time import perf_counter

from audit_kam_babilong_replay import QUESTIONS, TASKS, find_events, replay
from audit_kam_natural_data import digest, rank

WINDOW = 80
STRIDE = 60
BUDGETS = (4, 8, 16)


def chunks(text):
    words = tuple(re.finditer(r"\S+", text))
    bounds = [(words[i].start(), words[min(i + WINDOW, len(words)) - 1].end())
              for i in range(0, max(1, len(words) - (WINDOW - STRIDE)), STRIDE)]
    assert bounds and bounds[0][0] == words[0].start() and bounds[-1][1] == words[-1].end()
    spans = [text[lo:hi] for lo, hi in bounds]
    events = []
    for (lo, _), span in zip(bounds, spans, strict=True):
        events.append([(lo + start, lo + end, data) for start, end, data in find_events(span)])
    return spans, events


def unique_events(indices, events):
    selected = {}
    for i in indices:
        for lo, hi, item in events[i]:
            selected[lo, hi] = item
    return [(lo, hi, info) for (lo, hi), info in sorted(selected.items())]


def evaluate(case, picks, windows):
    events = unique_events(picks, windows)
    evidence = {(a, b) for a, b, _ in events}
    truth = {tuple(span) for span in case["evidence"]}
    found = len(evidence.intersection(truth))
    predicted, _ = replay(case["task"], case["question"], events)
    gold = case["gold"]
    if isinstance(gold, list):
        gold = tuple(gold)
    exact = predicted == gold
    return {"windows": picks, "events": len(evidence), "complete_witness": truth <= evidence,
            "witness_spans_recovered": found, "exact": exact, "joint": bool(truth) and truth <= evidence and exact}


def second_query(case, initial, events):
    if case["task"] not in ("qa2", "qa3"):
        return None
    object_ = QUESTIONS[case["task"]].fullmatch(case["question"].strip())["object"]
    for _, _, event in reversed(unique_events(initial, events)):
        if object_ in (event["take"], event["drop"], event["give"]):
            actor = event["receiver"] or event["actor"]
            return f"{case['question'].strip()} Where did {actor} go?"
    return None


def run(a, fixture):
    import torch

    from run_kam_natural_feasibility import Encoder

    torch.set_float32_matmul_precision("highest")
    device = torch.cuda.get_device_name()
    encoder = Encoder(a.assets / "encoder", 128)
    revisions = json.loads((a.assets / "revisions.json").read_text())
    torch.cuda.synchronize()
    begin = perf_counter()
    rows = []
    panel = [q for q in fixture["rows"] if q["task"] == a.task and q["length"] == a.length]
    assert len(panel) in (86, 100)
    for case in panel:
        spans, events = chunks(case["text"])
        total = len(spans)
        assert {(l, r) for l, r, _ in unique_events(range(total), events)} == {
            (l, r) for l, r, _ in find_events(case["text"])}
        eligible = {i for i, e in enumerate(events) if e}
        query = case["question"]
        first, = encoder([query], "query")
        started = perf_counter()
        bank = encoder(spans, "passage")
        scores = bank @ first
        ranking = scores.argsort(descending=True, stable=True).tolist()
        encode_seconds = perf_counter() - started
        lexical = rank(query, spans, limit=total)
        assert len(lexical) == len(ranking) == total and set(lexical) == set(range(total))
        compiled = {}
        for cap in BUDGETS:
            k = min(cap, total)
            arms = {"dense": ranking[:k], "lexical": lexical[:k],
                    "grammar_dense": [i for i in ranking if i in eligible][:k]}
            if a.task in ("qa2", "qa3"):
                initial = ranking[:max(1, k // 2)]
                conditioned = second_query(case, initial, events)
                if conditioned:
                    update = encoder([conditioned], "query")[0]
                    blended = ((bank @ update) + scores) / 2
                    remaining = blended.argsort(descending=True, stable=True).tolist()
                    arms["sequential"] = list(dict.fromkeys(initial + remaining))[:k]
                else:
                    arms["sequential"] = ranking[:k]
            compiled[str(cap)] = {name: evaluate(case, selected, events) for name, selected in arms.items()}
        rows.append({"id": case["id"], "windows": total, "event_windows": len(eligible),
                     "key_seconds": encode_seconds, "budgets": compiled})
    torch.cuda.synchronize()
    return {"fixture_sha256": digest(a.fixture), "task": a.task, "length": a.length, "device": device,
            "models": revisions, "elapsed_seconds": perf_counter() - begin, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("assets", "fixture", "output"):
        p.add_argument("--" + option, type=Path, required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--length", choices=("16k", "64k"), required=True)
    a = p.parse_args()
    with gzip.open(a.fixture, "rt") as stream:
        data = json.load(stream)
    answer = run(a, data)
    a.output.write_bytes(gzip.compress(json.dumps(answer, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
