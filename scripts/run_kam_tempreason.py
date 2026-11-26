#!/usr/bin/env python3
"""Test frozen retrieval and explicit temporal parsing on TempReason fact text."""

import argparse
from collections import Counter
from pathlib import Path
import re
from time import perf_counter

import torch

from audit_kam_natural_data import digest, words
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import Encoder, measure_answer

MONTHS = {month: i for i, month in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
DATE_TOKEN = r"([A-Z][a-z]{2}),?\s+([0-9]{1,4})"
INTERVAL = re.compile(rf"^(.*?)\s+from\s+{DATE_TOKEN}\s+to\s+{DATE_TOKEN}[.!]?\s*$")
POINT = re.compile(rf"\bin\s+{DATE_TOKEN}\s*[?]$")
RELATIVE = re.compile(r"\b(before|after)\s+(.+?)[?]$", flags=re.IGNORECASE)
SCOPE = {
    "P54": (" plays for ", 0),
    "P39": (" holds the position of ", 0),
    "P108": (" works for ", 0),
    "P6": (" is the head of the government of ", 1),
    "P488": (" is the chair of ", 1),
    "P286": (" is the head coach of ", 1),
    "P69": (" attended ", 0),
    "P102": (" is a member of ", 0),
    "P35": (" is the head of the state of ", 1),
    "P127": (" is owned by ", 0),
}


def norm(txt):
    return " " + " ".join(words(txt)) + " "


def month_index(month, year):
    return 12 * int(year) + MONTHS[month] if month in MONTHS else None


def parsed(fact):
    match = INTERVAL.fullmatch(fact)
    if match is None:
        return None
    text, m1, y1, m2, y2 = match.groups()
    lo, hi = month_index(m1, y1), month_index(m2, y2)
    if lo is None or hi is None or hi < lo:
        return None
    return text, lo, hi


def scope_matches(text, question, prop):
    marker, side = SCOPE[prop]
    parts = text.split(marker, 1)
    return len(parts) == 2 and norm(parts[side]) in norm(question)


def rule_candidates(q, facts, prop, level, scoped):
    parsed_facts = [(i, x) for i, fact in enumerate(facts) if (x := parsed(fact))]
    eligible = [(i, fact) for i, fact in parsed_facts if not scoped or scope_matches(fact[0], q, prop)]
    if level == 2:
        match = POINT.search(q)
        date = month_index(*match.groups()) if match else None
        if date is None:
            return []
        return [i for i, (_, lo, hi) in eligible if lo <= date <= hi]
    match = RELATIVE.search(q)
    if match is None:
        return []
    direction, name = match.groups()
    anchors = [lo for _, (text, lo, _) in eligible if norm(name) in norm(text)]
    if len(set(anchors)) != 1:
        return []
    anchor = anchors[0]
    options = [(i, lo) for i, (_, lo, _) in eligible
               if (lo > anchor if direction.casefold() == "after" else lo < anchor)]
    if not options:
        return []
    point = (min if direction.casefold() == "after" else max)(lo for _, lo in options)
    return [i for i, lo in options if lo == point]


def report_evidence(selected, gold_sets):
    flags = [bool(set(selected) & supports) for supports in gold_sets]
    return {"at_least_one": any(flags), "all": all(flags), "returned": bool(selected)}


@torch.inference_mode()
def evaluate(a, fixture, rows):
    encoder = Encoder(a.assets / "encoder", 64)
    reader = Reader(a.assets / "reader")
    total_cases = [r["facts"] + r["noise"] for r in rows]
    lengths = [len(facts) for facts in total_cases]
    bank = encoder([fact for docs in total_cases for fact in docs], "passage")
    queries = encoder([r["question"] for r in rows], "query")
    records, requests = [], []
    for query, keys, r, all_facts in zip(queries, bank.split(lengths), rows, total_cases, strict=True):
        record = {"id": r["id"], "entity": r["entity"], "property": r["property"],
                  "parsed_facts": sum(parsed(f) is not None for f in r["facts"]),
                  "n_facts": len(r["facts"]), "scenarios": {}}
        gold_sets = [{i for i, fact in enumerate(r["facts"]) if norm(answer) in norm(fact)} for answer in r["answers"]]
        record["gold_sets"] = [sorted(slots) for slots in gold_sets]
        full = query @ keys.T
        for scenario, count in (("clean", len(r["facts"])), ("noise", len(all_facts))):
            facts = all_facts[:count]
            dense = full[:count].argsort(descending=True, stable=True).tolist()
            selectors = {"dense_top1": dense[:1], "dense_top2": dense[:2],
                         "explicit_time": rule_candidates(r["question"], facts, r["property"], fixture["level"], False),
                         "explicit_time_and_scope": rule_candidates(r["question"], facts,
                                                                    r["property"], fixture["level"], True)}
            for arm in ("explicit_time", "explicit_time_and_scope"):
                options = set(selectors[arm])
                selectors[arm] = [slot for slot in dense if slot in options][:2]
            outputs = {}
            for name, ids in selectors.items():
                info = {"slots": ids, **report_evidence(ids, gold_sets), "noise_slots": sum(i >= len(r["facts"]) for i in ids)}
                outputs[name] = info
                ex = {"question": r["question"], "context": [("Event", [f]) for f in facts]}
                requests.append((reader.prompt(ex, ids), info, r["answers"]))
            record["scenarios"][scenario] = outputs
        gold = sorted({slot for group in gold_sets for slot in group})[:2]
        record["oracle"] = report_evidence(gold, gold_sets)
        ex = {"question": r["question"], "context": [("Event", [f]) for f in r["facts"]]}
        requests.append((reader.prompt(ex, gold), record["oracle"], r["answers"]))
        records.append(record)
    prompts = list(dict.fromkeys(p for p, _, _ in requests))
    answers = dict(zip(prompts, reader.answers(prompts), strict=True))
    for p, target, truths in requests:
        target["em"] = max(measure_answer(answers[p], gold)["em"] for gold in truths)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "fixture", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    data = read(a.fixture)
    rows = evaluate(a, data, data["rows"][a.shard::a.shards])
    save(a.output, {"level": data["level"], "fixture_sha256": digest(a.fixture), "source_sha256": data["source_sha256"],
                    "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                    "rows": rows, "elapsed_seconds": perf_counter() - start})


if __name__ == "__main__":
    main()
