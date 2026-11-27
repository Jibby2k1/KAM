#!/usr/bin/env python3
"""Verify a chronological state oracle on pinned BABILong object tasks."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

from audit_kam_natural_data import digest, stable

REVISION = "ee0d588794c7ac098062ee0d247c733d62e94fe2"
TASKS = ("qa2", "qa3", "qa7", "qa8")
SCOPES = ("0k", "16k", "64k")
ACTOR = "Mary|John|Daniel|Sandra"
LOCATION = "bathroom|bedroom|garden|hallway|kitchen|office"
ITEM = "apple|football|milk"
MOVE = "went back|went|journeyed|moved|travelled"
TAKE = "got|grabbed|picked up|took"
DROP = "discarded|dropped|left|put down"
GIVE = "gave|handed|passed"
EVENT = re.compile(
    rf"\b(?P<actor>{ACTOR}) (?:(?:{MOVE}) to the (?P<place>{LOCATION})|"
    rf"(?:{TAKE}) the (?P<take>{ITEM})(?: there)?|(?:{DROP}) the (?P<drop>{ITEM})(?: there)?|"
    rf"(?:{GIVE}) the (?P<give>{ITEM}) to (?P<receiver>{ACTOR}))(?:\.)"
)
QUESTIONS = {
    "qa2": re.compile(rf"Where is the (?P<object>{ITEM})\?"),
    "qa3": re.compile(rf"Where was the (?P<object>{ITEM}) before the (?P<destination>{LOCATION})\?"),
    "qa7": re.compile(rf"How many objects is (?P<owner>{ACTOR}) carrying\?"),
    "qa8": re.compile(rf"What is (?P<owner>{ACTOR}) carrying\?"),
}


def find_events(text):
    return [(match.start(), match.end(), match.groupdict()) for match in EVENT.finditer(text)]


def replay(task, question, events):
    query = QUESTIONS[task].fullmatch(question.strip())
    assert query is not None, (task, question)
    places, owners, paths, ownership_updates = {}, {}, {}, {}

    def record(item, actor, i):
        if actor not in places:
            return
        place, when = places[actor]
        path = paths.setdefault(item, [])
        proof = ({i, when} if when != i else {i, ownership_updates[item]})
        if path and path[-1][0] == place:
            path[-1] = place, proof
        else:
            path.append((place, proof))

    for i, (_, _, fields) in enumerate(events):
        actor = fields["actor"]
        if place := fields["place"]:
            places[actor] = place, i
            for item, owner in owners.items():
                if owner == actor:
                    record(item, actor, i)
            continue
        item = fields["take"] or fields["drop"] or fields["give"]
        ownership_updates[item] = i
        if fields["drop"]:
            record(item, actor, i)
            owners[item] = None
        else:
            owner = fields["receiver"] or actor
            owners[item] = owner
            record(item, actor, i)
    if task == "qa2":
        path = paths.get(query["object"], [])
        return path[-1] if path else (None, set())
    if task == "qa3":
        path = paths.get(query["object"], [])
        for (place, evidence), (destination, entry) in reversed(list(zip(path, path[1:]))):
            if destination == query["destination"]:
                return place, evidence | entry
        return None, set()

    current = sorted(k for k, v in owners.items() if v == query["owner"])
    # For empty states, expose the transitions which removed items from the
    # owner as well as transitions that put currently carried items there.
    evidence = {i for obj, i in ownership_updates.items()
                if query["owner"] in (events[i][2]["actor"], events[i][2]["receiver"])}
    return (len(current) if task == "qa7" else tuple(current)), evidence


def normalized(task, answer):
    if task == "qa7":
        return {"none": 0, "one": 1, "two": 2, "three": 3}[answer]
    if task == "qa8":
        return tuple(sorted(answer.split(","))) if answer != "nothing" else ()
    return answer


def signatures(events):
    return [e[2] for e in events]


def prepare(inputs):
    verified, public = [], []
    explanations = {}
    for task in TASKS:
        files = {scope: inputs / f"{task}-{scope}.json" for scope in SCOPES}
        data = {k: json.loads(p.read_text()) for k, p in files.items()}
        dimensions = {}
        for scope in SCOPES:
            counts = Counter()
            source_cases = data[scope]
            assert len(source_cases) == len(data["0k"]) == 100
            for ordinal, (q, first) in enumerate(zip(source_cases, data["0k"], strict=True)):
                text = q["input"]
                events = find_events(text)
                matches_clean = signatures(events) == signatures(find_events(first["input"]))
                same_query = (q["question"], q["target"]) == (first["question"], first["target"])
                predicted, evidence = replay(task, q["question"], events)
                gold = normalized(task, q["target"])
                counts.update({"n": 1, "same_question_and_label_by_row": same_query,
                               "same_extracted_events_by_row": matches_clean,
                               "paired": same_query and matches_clean,
                               "exact": predicted == gold, "nonempty_proof": bool(evidence),
                               "exact_positive_inventory": task in ("qa7", "qa8") and bool(gold) and predicted == gold,
                               "positive_inventory": task in ("qa7", "qa8") and bool(gold)})
                if scope == "0k":
                    assert len(text.split(".")) - 1 == len(events)
                if predicted != gold or not (same_query and matches_clean):
                    continue
                id_ = stable(f"babilong:{REVISION}:{task}:{ordinal}")[:20]
                entry = {"id": id_, "task": task, "length": scope, "ordinal": ordinal, "input_chars": len(text),
                         "gold": list(gold) if isinstance(gold, tuple) else gold,
                         "evidence": [[events[i][0], events[i][1]] for i in sorted(evidence)],
                         "event_count": len(events)}
                public.append(entry)
                verified.append(entry | {"text": text, "question": q["question"]})
            dimensions[scope] = {"sha256": digest(files[scope]), "audit": counts}
        explanations[task] = dimensions
    provenance = {"source": "https://huggingface.co/datasets/RMT-team/babilong", "revision": REVISION,
                  "tasks": explanations}
    return provenance, verified, public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "dataset", "manifest", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    meta, fixtures, pure = prepare(a.inputs)
    a.dataset.write_bytes(gzip.compress(json.dumps(meta | {"rows": fixtures}, separators=(",", ":")).encode(), mtime=0))
    manifest = meta | {"fixture_sha256": digest(a.dataset), "rows": pure}
    a.manifest.write_bytes(gzip.compress(json.dumps(manifest, separators=(",", ":")).encode(), mtime=0))
    a.output.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
