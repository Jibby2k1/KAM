#!/usr/bin/env python3
"""Join published TimeQA human variants to exact annotated source paragraphs."""

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_kam_natural_data import digest, words
from prepare_kam_timeqa_contrasts import windows


def normalize(value):
    return " ".join(words(value))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("easy", "hard", "annotated", "prior", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    easy, hard = ({q["idx"]: q for line in file.read_text().splitlines() if line for q in [json.loads(line)]}
                  for file in (a.easy, a.hard))
    assert easy.keys() == hard.keys()
    source = {row["index"]: row for row in json.loads(a.annotated.read_text())}
    training = {row["entity"] for row in json.loads(a.prior.read_text())["rows"]["dev"]}
    counts = Counter(total=len(easy))
    selected = []
    for id_, original in easy.items():
        note, index = id_.rsplit("#", 1)
        doc = source[note]
        if doc["link"] in training:
            counts["fit_entity_overlap"] += 1
            continue
        difficulty = hard[id_]
        assert original["context"] == difficulty["context"]
        assert sorted(original["targets"]) == sorted(difficulty["targets"])
        targets = [t for t in original["targets"] if normalize(t)]
        _, spans = doc["questions"][int(index)]
        gold_by_para = [(r["para"], normalize(r["answer"])) for r in spans if normalize(r["answer"])]
        if not targets or not gold_by_para:
            counts["no_answer"] += 1
            continue
        assert {normalize(x) for x in targets} == {answer for _, answer in gold_by_para}
        contexts = [(ix, offset, text) for ix, body in enumerate(doc["paras"]) if ix
                    for offset, text in windows(body)]
        gold = [idx for idx, (ix, _, text) in enumerate(contexts) if any(
            ix == para and f" {answer} " in f" {normalize(text)} " for para, answer in gold_by_para)]
        if not gold:
            counts["no_visible_span"] += 1
            continue
        selected.append({"id": id_, "entity": doc["link"], "kind": doc["type"], "title": doc["paras"][0],
                         "contexts": contexts, "targets": targets, "gold": gold,
                         "easy": original["question"], "hard": difficulty["question"]})
        counts["selected"] += 1
        counts["edited_question"] += original["question"] != difficulty["question"]
    assert len(selected) >= 600
    data = {"sources": {name: digest(getattr(a, name)) for name in ("easy", "hard", "annotated", "prior")},
            "counts": dict(counts), "rows": selected}
    a.output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(data["counts"])


if __name__ == "__main__":
    main()
