#!/usr/bin/env python3
"""Development-only addressability and prior-turn feasibility checks."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile


def words(text):
    return re.findall(r"\w+", text.casefold())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(id_):
    return hashlib.sha256(id_.encode()).hexdigest()


def rank(query, documents, limit=2):
    tokens = words(query)
    counts = [Counter(words(doc)) for doc in documents]
    frequency = Counter(word for document in counts for word in document)
    lengths = [sum(document.values()) for document in counts]
    average = sum(lengths) / len(lengths) or 1
    weights = {word: math.log1p((len(counts) - frequency[word] + .5) / (frequency[word] + .5))
               for word in set(tokens)}
    scores = [sum(idf * (tf := count[word]) * 2.2 / (tf + 1.2 * (.25 + .75 * n / average))
                  for word, idf in weights.items())
              for count, n in zip(counts, lengths)]
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:limit]


def hotpot(path, limit):
    items = json.loads(path.read_text())
    eligible = []
    counts = Counter()
    for item in items:
        if item["type"] != "bridge":
            continue
        counts["bridge"] += 1
        context = dict(item["context"])
        title_set = {title for title, _ in item["supporting_facts"]}
        if len(title_set) != 2 or not all(title in context and 0 <= index < len(context[title])
                                        for title, index in item["supporting_facts"]):
            continue
        counts["valid_two_document_gold"] += 1
        eligible.append(item)
    sample = sorted(eligible, key=lambda item: stable(item["_id"]))[:limit]
    result = []
    for item in sample:
        titles = [title for title, _ in item["context"]]
        gold = {titles.index(title) for title, _ in item["supporting_facts"]}
        documents = [title + " " + " ".join(sentences) for title, sentences in item["context"]]
        ranked = rank(item["question"], documents)
        gold_text = " ".join(dict(item["context"])[title][index] for title, index in item["supporting_facts"])
        has_answer = " " + " ".join(words(item["answer"])) + " " in " " + " ".join(words(gold_text)) + " "
        result.append({"id": item["_id"], "gold": sorted(gold), "bm25": ranked,
                       "both_retrieved": gold <= set(ranked), "answer_string_in_gold_sentences": has_answer})
    return {"source_sha256": digest(path), "scanned": len(items), "counts": counts, "sample": result}


def dialogue(path, limit):
    with ZipFile(path) as archive:
        source = "multidoc2dial/multidoc2dial_dial_validation.json"
        dialogues = json.loads(archive.read(source))["dial_data"]
    counts = Counter()
    examples = []
    for domain, conversations in sorted(dialogues.items()):
        for conv in conversations:
            counts["conversations"] += 1
            history = []
            prompts = []
            for turn in conv["turns"]:
                if turn["role"] == "user":
                    prompts.append(turn["utterance"])
                    continue
                refs = {ref["doc_id"] for ref in turn["references"]}
                if not prompts or len(refs) != 1:
                    counts["excluded_agent_turns"] += 1
                    continue
                doc = next(iter(refs))
                matches = [i for i, past in enumerate(history) if past[0] == doc]
                if matches and history[-1][0] != doc:
                    counts["document_return"] += 1
                    recent = history[-2:]
                    examples.append({"id": f"{conv['dial_id']}:{turn['turn_id']}", "domain": domain,
                                     "eligible_turns_ago": len(history) - matches[-1],
                                     "in_last_two": any(past[0] == doc for past in recent),
                                     # Retrieval sees past utterances. Prior annotations
                                     # are used only to score which past turn it chose.
                                     "prior_utterance_bm25_hit": history[rank(prompts[-1],
                                         [past[1] for past in history], limit=1)[0]][0] == doc})
                elif matches:
                    counts["same_document_continuation"] += 1
                else:
                    counts["first_visit"] += 1
                history.append((doc, turn["utterance"]))
    return {"source_sha256": digest(path), "member": source, "counts": counts,
            "sample": sorted(examples, key=lambda row: stable(row["id"]))[:limit]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=("hotpot", "dialogue"))
    parser.add_argument("source", type=Path)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"track": args.track, "split": "development", **globals()[args.track](args.source, args.limit)}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
