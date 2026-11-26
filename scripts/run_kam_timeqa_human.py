#!/usr/bin/env python3
"""Transfer a frozen date-aware selector to paired published TimeQA human questions."""

import argparse
from collections import Counter
from pathlib import Path
import re
from time import perf_counter

import torch
from torch import nn

from audit_kam_natural_data import digest
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import Encoder, measure_answer
from run_kam_timeqa_contrasts import data_matrix, transform

DECADE = re.compile(r"\b(?:(early|mid|late)\s+)?((?:1[0-9]|20)\d0)s\b", flags=re.IGNORECASE)
YEAR = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
MONTH = re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
                   r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*(?=[0-9]{4}\b)", re.IGNORECASE)


def dates(question):
    year_hits = list(YEAR.finditer(question))
    decade = DECADE.search(question)
    if year_hits:
        points = [int(m[0]) for m in year_hits]
        if len(set(points)) > 1:
            return "range", min(points), max(points)
        anchor = points[0]
        lead = question[max(0, year_hits[0].start() - 28):year_hits[0].start()].casefold()
        if re.search(r"\b(before|prior to|until|by)\b", lead):
            return "before", 1500, anchor
        if re.search(r"\b(after|since|following)\b", lead):
            return "after", anchor, 2200
        return "year", anchor, anchor
    if decade:
        modifier, start = (decade[1] or "").casefold(), int(decade[2])
        a, b = {"early": (0, 3), "mid": (3, 6), "late": (7, 9)}.get(modifier, (0, 9))
        return "decade", start + a, start + b
    return "unparsed", 1500, 2200


def make_matrix_entries(rows):
    matrix, tags = [], []
    for row in rows:
        queries, kinds = [], {}
        for arm in ("easy", "hard"):
            question = row[arm]
            kind, lower, upper = dates(question)
            query_undated = YEAR.sub("", MONTH.sub("", DECADE.sub("", question)))
            queries.append({"query": question, "query_without_date": query_undated,
                            "lower": lower * 12, "upper": upper * 12 + 11})
            kinds[arm] = kind
        matrix.append({"title": row["title"], "context": row["contexts"], "noise": [], "questions": queries})
        tags.append(kinds)
    return matrix, tags


@torch.inference_mode()
def evaluate(a, cases, weights):
    encoder = Encoder(a.assets / "encoder", 64)
    reader = Reader(a.assets / "reader")
    rows, tags = make_matrix_entries(cases)
    matrices, contexts = data_matrix(encoder, rows)
    models = {}
    for name, state in weights["heads"].items():
        width = state["0.weight"].shape[1]
        head = nn.Sequential(nn.Linear(width, 16), nn.SiLU(), nn.Linear(16, 1)).cuda().eval()
        head.load_state_dict(state)
        models[name] = head, width
    results, requests = [], []
    for i, (r, context, kinds) in enumerate(zip(cases, contexts, tags, strict=True)):
        outcomes = {}
        for n, mode in enumerate(("easy", "hard")):
            matrix = matrices[2 * i + n]
            raw = int(matrix[:, 0].argmax())
            picks = {"dense": raw, "dense_no_date": int(matrix[:, 1].argmax()),
                     "oracle": min(r["gold"])}
            for name, (head, width) in models.items():
                picks[name] = int(head(transform(matrix, width)).argmax())
            if kinds[mode] == "unparsed":
                picks["learned_dates"] = raw
            scores = {}
            for name, picked in picks.items():
                item = {"slot": picked, "support": picked in r["gold"]}
                scores[name] = item
                prompt = reader.prompt({"question": r[mode], **context}, [picked])
                requests.append((prompt, item, r["targets"]))
            outcomes[mode] = {"date_rule": kinds[mode], "arms": scores}
        results.append({"id": r["id"], "entity": r["entity"], "kind": r["kind"],
                        "gold": r["gold"], "edited": r["easy"] != r["hard"], "outcomes": outcomes})
    prompts = list(dict.fromkeys(p for p, _, _ in requests))
    responses = dict(zip(prompts, reader.answers(prompts), strict=True))
    for prompt, item, truths in requests:
        measures = [measure_answer(responses[prompt], truth) for truth in truths]
        item.update(em=max(r["em"] for r in measures), f1=max(r["f1"] for r in measures))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "fixture", "training-fixture", "frozen", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    data = read(a.fixture)
    frozen = torch.load(a.frozen, weights_only=True, map_location="cpu")
    assert frozen["fixture_sha256"] == digest(a.training_fixture) == data["sources"]["prior"]
    assert frozen["models"] == read(a.assets / "revisions.json")
    rows = data["rows"][a.shard::a.shards]
    results = evaluate(a, rows, frozen)
    save(a.output, {"fixture_sha256": digest(a.fixture), "weights_sha256": digest(a.frozen),
                    "device": torch.cuda.get_device_name(), "recruitment": data["counts"],
                    "date_rules": {name: dict(Counter(r["outcomes"][name]["date_rule"] for r in results))
                                   for name in ("easy", "hard")},
                    "elapsed_seconds": perf_counter() - start, "rows": results})


if __name__ == "__main__":
    main()
