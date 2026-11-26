#!/usr/bin/env python3
"""Freeze two-period TimeQA contrasts and time-matched hard distractors."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from urllib.parse import unquote

from audit_kam_natural_data import digest, stable, words

RELATIONS = {
    "P54": "Which sports team did {person} play for",
    "P39": "Which position did {person} hold",
    "P108": "Who was {person}'s employer",
    "P69": "Which educational institution did {person} attend",
    "P26": "Who was {person}'s spouse",
    "P463": "Which organization was {person} a member of",
}
MONTHS = {month: i for i, month in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def endpoint(s, *, end=False):
    match = re.fullmatch(r"(?:(\w{3}) )?(\d{4})", s)
    if match is None or (match[1] and match[1] not in MONTHS):
        return None
    return 12 * int(match[2]) + MONTHS.get(match[1], 11 if end else 0)


def normalize(s):
    return " ".join(words(s))


def interval_rows(row):
    for period, spans in row["questions"]:
        lower = endpoint(period[0])
        upper = endpoint(period[1], end=True)
        if lower is None or upper is None or upper < lower:
            continue
        active = [span for span in spans if span["answer"]]
        if len(active) != 1:
            continue
        span = active[0]
        paragraph = row["paras"][span["para"]]
        if paragraph[span["from"]:span["end"]] != span["answer"]:
            continue
        # Keep lexical input within the GPU encoder's standard token budget
        # while verifying that the labeled substring is still retained.
        word_spans = list(re.finditer(r"\S+", paragraph))
        choices = []
        for start in range(0, len(word_spans), 60):
            window = word_spans[start:start + 96]
            if window[0].start() <= span["from"] and window[-1].end() >= span["end"]:
                choices.append(start)
        if choices:
            yield {"period": period, "lower": lower, "upper": upper, "answer": span["answer"],
                   "para": span["para"], "window_start": choices[0]}


def windows(paragraph):
    tokens = paragraph.split()
    for start in range(0, len(tokens), 60):
        yield start, " ".join(tokens[start:start + 96])


def contrast(rows, label, maximum, banned):
    counts = Counter(raw_entities=len({r["link"] for r in rows}), raw_rows=len(rows))
    eligible = []
    for row in rows:
        if row["type"] not in RELATIONS or row["link"] in banned:
            continue
        intervals = list(interval_rows(row))
        for i, first in enumerate(intervals):
            for second in intervals[i + 1:]:
                if (first["upper"] < second["lower"] and first["para"] != second["para"]
                        and normalize(first["answer"]) != normalize(second["answer"])):
                    eligible.append((row, first, second))
    counts["eligible_pairs"] = len(eligible)
    pool = defaultdict(list)
    for row, first, second in eligible:
        for event in (first, second):
            text = dict(windows(row["paras"][event["para"]]))[event["window_start"]]
            pool[row["type"]].append((row["link"], event, text))
    picked = []
    seen = set()
    order = sorted(eligible, key=lambda item: stable(
        f"kam-timeqa-contrast-v1:{label}:{item[0]['index']}:{item[1]['lower']}:{item[2]['lower']}"))
    for row, first, second in order:
        if row["link"] in seen:
            continue
        person = unquote(row["link"].removeprefix("/wiki/")).replace("_", " ")
        context = [(i, start, text) for i, p in enumerate(row["paras"]) if i > 0
                   for start, text in windows(p)]
        if len(context) < 2:
            continue
        keys = {(i, start): j for j, (i, start, _) in enumerate(context)}
        alternatives = {normalize(first["answer"]), normalize(second["answer"])}
        person_key = normalize(person.split(" (", 1)[0])
        noise = []
        donor_links = {row["link"]}
        donor_texts = set()
        for event in (first, second):
            candidates = sorted(pool[row["type"]], key=lambda d: (
                max(0, event["lower"] - d[1]["upper"], d[1]["lower"] - event["upper"]),
                stable(row["index"] + d[0] + str(d[1]["lower"]))))
            tranche = 0
            for donor, fact, passage in candidates:
                text = normalize(passage)
                if donor in donor_links or text in donor_texts or person_key in text:
                    continue
                if any(answer in text for answer in alternatives):
                    continue
                donor_links.add(donor)
                donor_texts.add(text)
                noise.append({"entity": donor, "period": fact["period"], "text": passage})
                tranche += 1
                if tranche == 4:
                    break
        if len(noise) != 8:
            counts["insufficient_donors"] += 1
            continue
        anchor = RELATIONS[row["type"]].format(person=person)
        questions = []
        for event in (first, second):
            answer = normalize(event["answer"])
            gold = [j for j, (i, _, text) in enumerate(context)
                    if i == event["para"] and f" {answer} " in f" {normalize(text)} "]
            assert keys[(event["para"], event["window_start"])] in gold
            questions.append({**event, "gold": gold, "query_without_date": anchor + "?",
                              "query": f"{anchor} from {event['period'][0]} to {event['period'][1]}?"})
        picked.append({"id": row["index"], "entity": row["link"], "type": row["type"],
                       "title": person, "questions": questions, "context": context, "noise": noise})
        seen.add(row["link"])
        if len(picked) == maximum:
            break
    counts["selected_entities"] = len(picked)
    return picked, dict(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for item in ("dev", "test", "output"):
        p.add_argument("--" + item, type=Path, required=True)
    p.add_argument("--max-entities", type=int, default=320)
    a = p.parse_args()
    dev, test = [json.loads(path.read_text()) for path in (a.dev, a.test)]
    test_entities = {r["link"] for r in test}
    dev_cases, dev_counts = contrast(dev, "development", a.max_entities, test_entities)
    test_cases, test_counts = contrast(test, "test", a.max_entities, set())
    assert len(dev_cases) >= 80 and len(test_cases) >= 80
    assert not {r["entity"] for r in dev_cases} & {r["entity"] for r in test_cases}
    data = {"source": "TimeQA crowd-verified interval annotations", "dev_sha256": digest(a.dev),
            "test_sha256": digest(a.test), "counts": {"dev": dev_counts, "test": test_counts},
            "rows": {"dev": dev_cases, "test": test_cases}}
    a.output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(data["counts"])


if __name__ == "__main__":
    main()
