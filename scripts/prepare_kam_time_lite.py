#!/usr/bin/env python3
"""Freeze the author-released TIME-Lite multiple-choice subset and context windows."""

import argparse
from collections import Counter
from datetime import datetime
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re

REVISION = "cd93b48ffeb8491f6d46359a8e2bf79495cb0ad3"
SOURCE_SHA = "6643fc42c7048fa484469ebab602ce13fd6988fb689fed90ebd7774202b8b551"
SETTINGS = {"TIME-Lite-News": "hybrid", "TIME-Lite-Dial": "base", "TIME-Lite-Wiki": "base"}
OPTION = re.compile(r"(?m)^[ \t]*([A-D])[.)][ \t]*")
NEWS = re.compile(r"(?m)^Title: (.+), Day: ([^\n]+)\nContent: ")
DIAL = re.compile(r"(?m)^Session (\d+) happened at ([^\n]+)\.[ \t]*\n")
WORD_CAP = 140
WORD_STEP = 112


def windows(body):
    words = body.split()
    for start in range(0, len(words), WORD_STEP):
        end = min(start + WORD_CAP, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break


def contexts(text, domain):
    if domain == "TIME-Lite-Wiki":
        docs = [("", "", text)]
    else:
        news = domain == "TIME-Lite-News"
        matches = list((NEWS if news else DIAL).finditer(text))
        expected = 3 if news else len(re.findall(r"(?m)^Session \d+ happened at ", text))
        assert len(matches) == expected and expected > 1
        docs = []
        for i, match in enumerate(matches):
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            label, calendar = match.groups()
            if news:
                stamp = datetime.strptime(calendar, "%B %d, %Y").date().isoformat()
            elif " on " in calendar:
                stamp = datetime.strptime(calendar, "%I:%M %p on %d %B, %Y").isoformat(sep=" ", timespec="minutes")
            else:
                stamp = datetime.strptime(calendar, "%d.%m.%Y, %H:%M:%S").isoformat(sep=" ", timespec="seconds")
            docs.append((label if news else "", stamp, text[match.end():end]))
    result = []
    for record, (title, day, text) in enumerate(docs):
        for piece in windows(text):
            result.append({"body": f"{title}. {piece}" if title else piece, "day": day, "source_index": record})
    assert result
    return result


def question(row):
    raw, label = row["Question"], row["Gold Answer"]
    if label not in {"A", "B", "C", "D"}:
        return None
    matches = list(OPTION.finditer(raw))
    labels = [m[1] for m in matches]
    assert labels == list("ABCD"[:len(matches)]) and label in labels and len(labels) in (3, 4)
    texts = [raw[m.end() : matches[i+1].start() if i+1 < len(matches) else len(raw)].strip()
             for i, m in enumerate(matches)]
    assert len(set(texts)) == len(texts) and all(texts)
    return {"question": raw[:matches[0].start()].strip(), "options": texts, "gold": labels.index(label)}


def seal(source):
    payload = source.read_bytes()
    assert sha256(payload).hexdigest() == SOURCE_SHA
    rows = json.loads(payload)
    by_domain = {d: {"questions": [], "contexts": []} for d in SETTINGS}
    cache = {d: {} for d in SETTINGS}
    for row in rows:
        d = row["Dataset Name"]
        if row["Setting"] != SETTINGS[d]:
            continue
        query = question(row)
        if query is None:
            continue
        stamp = sha256(row["Question"].encode()).hexdigest()
        text = row["Context"]
        context_id = sha256(text.encode()).hexdigest()
        domain = by_domain[d]
        if context_id not in cache[d]:
            cache[d][context_id] = len(domain["contexts"])
            domain["contexts"].append({"sha256": context_id, "windows": contexts(text, d)})
        domain["questions"].append(query | {"id": stamp, "family": row["Task"], "level": row["Level"],
                                           "context": cache[d][context_id]})
    manifest = {"source_revision": REVISION, "source_sha256": SOURCE_SHA, "window_words": WORD_CAP, "step_words": WORD_STEP,
                "domains": {}}
    for d, domain in by_domain.items():
        cases, context = domain["questions"], domain["contexts"]
        ids = {q["id"] for q in cases}
        assert len(ids) == len(cases)
        source_groups = []
        for ctx in context:
            documents = {}
            for window in ctx["windows"]:
                documents.setdefault(window["source_index"], []).append((window["day"], window["body"]))
            digests = [sha256(json.dumps(v, separators=(",", ":")).encode()).hexdigest() for v in documents.values()]
            source_groups.append(digests)
        manifest["domains"][d] = {"source_groups": source_groups, "questions": len(cases), "contexts": len(context), "source_setting": SETTINGS[d],
                                    "choice_counts": dict(Counter(len(q["options"]) for q in cases)),
                                    "task_counts": dict(Counter(q["family"] for q in cases)),
                                    "windows": sum(len(x["windows"]) for x in context),
                                    "rows": [{k: v for k, v in q.items() if k in ("id", "family", "level", "gold", "context")}
                                             | {"options": len(q["options"])} for q in cases]}
    return by_domain, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "directory"):
        p.add_argument("--" + k, required=True, type=Path)
    a = p.parse_args()
    by_domain, manifest = seal(a.source)
    a.directory.mkdir(parents=True, exist_ok=True)
    for d, data in by_domain.items():
        target = a.directory / f"{d}.json.gz"
        target.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0))
        manifest["domains"][d]["fixture_sha256"] = sha256(target.read_bytes()).hexdigest()
    (a.directory/"manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
