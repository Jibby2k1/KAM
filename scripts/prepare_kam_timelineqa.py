#!/usr/bin/env python3
"""Build a temporal provenance probe from seeded author-generated TimelineQA episodes."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import re


DATE = re.compile(r"\b\d{4}/\d{2}/\d{2}\b")
CATEGORIES = ("exercise", "lunch", "read", "watch tv")
SCHEMA = "timelineqa_episode_provenance_v1"


def words(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def prepare(seed, density, path):
    original = path.read_bytes()
    logs = json.loads(original)
    indexed = {}
    keys = defaultdict(list)
    seeds = []
    for date, entries in sorted(logs.items()):
        assert DATE.fullmatch(date)
        for key, event in sorted(entries.items()):
            eid = event["eid"]
            assert eid not in indexed
            text = event["text_template_based"]
            if not isinstance(text, str) or not text.strip():
                continue
            kind = re.sub(r"\d+$", "", key)
            indexed[eid] = {"day": date, "type": kind, "text": text}
            for q, a in event["atomic_qa_pairs"]:
                if kind not in CATEGORIES or not isinstance(a, str) or not a.strip():
                    continue
                explicit = DATE.findall(q)
                if explicit != [date] or DATE.search(a) or a.casefold() in ("yes", "no"):
                    continue
                phrase = words(DATE.sub(" DATE ", q))
                answer = " " + words(a) + " "
                if answer not in " " + words(text) + " ":
                    continue
                item = (eid, q, a, phrase)
                keys[kind, phrase].append(item)
                seeds.append(item)
    rng = Random(seed + 119268)
    rng.shuffle(seeds)
    selected = defaultdict(list)
    episode_ids = list(indexed)
    used_events = set()
    for eid, question, answer, phrase in seeds:
        info = indexed[eid]
        day, kind = info["day"], info["type"]
        if eid in used_events or len(selected[kind]) == 6:
            continue
        alternatives = [x[0] for x in keys[kind, phrase] if indexed[x[0]]["day"] != day and
                        " " + words(answer) + " " not in " " + words(indexed[x[0]]["text"]) + " "]
        if not alternatives:
            continue
        # Retain all same-day records in the shared minimum candidate pool.
        same_day = [x for x, v in indexed.items() if v["day"] == day]
        if len(same_day) > 14:
            continue
        donor = rng.choice(sorted(set(alternatives)))
        reserve = set(same_day) | {donor}
        chosen = sorted(reserve) + rng.sample([x for x in episode_ids if x not in reserve], 256 - len(reserve))
        priority = list(range(256))
        rng.shuffle(priority)
        selected[kind].append({"id": f"{seed}:{eid}", "gold": eid, "donor": donor, "candidates": chosen,
                               "shuffle": priority, "question": question, "answer": answer})
        used_events.add(eid)
    assert all(len(selected[k]) == 6 for k in CATEGORIES), (seed, {k: len(v) for k, v in selected.items()})
    flat = [case for kind in CATEGORIES for case in selected[kind]]
    used = {eid for case in flat for eid in case["candidates"]}
    return {"seed": seed, "density": density, "original_sha256": sha256(original).hexdigest(),
            "original_episodes": len(indexed), "episodes": {k: indexed[k] for k in sorted(used)}, "cases": flat}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--generator-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    cfg = p.parse_args()
    personas = [prepare(int(seed), density, cfg.episodes / seed / "episodes.json")
                for seed, density in (line.split() for line in cfg.manifest.read_text().splitlines())]
    assert len({p["seed"] for p in personas}) == len(personas)
    output = {"schema": SCHEMA, "author_generator_revision": cfg.generator_revision, "personas": personas}
    cfg.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
