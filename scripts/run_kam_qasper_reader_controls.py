#!/usr/bin/env python3
"""Audit paper-title conditioning, answer visibility, and reader capacity on QASPER."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest, words
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import Encoder, candidate_texts, measure_answer
from run_kam_qasper_evidence import evidence_metrics


def plain_question(row):
    prefix = "In the paper " + row["context"][0][0].rsplit(" — ", 1)[0] + ", "
    assert row["question"].startswith(prefix)
    return row["question"][len(prefix):]


@torch.inference_mode()
def select(rows, encoder):
    contexts = [r["context"] + r["noise"] for r in rows]
    sizes = list(map(len, contexts))
    bank = encoder([x for ctx in contexts for x in candidate_texts({"context": ctx})], "passage").split(sizes)
    unprefixed = encoder([plain_question(r) for r in rows], "query")
    prefixed = encoder([r["question"] for r in rows], "query")
    values = []
    for j, row in enumerate(rows):
        case = {}
        for scenario, size in (("clean", len(row["context"])), ("noise", sizes[j])):
            case[scenario] = {}
            for policy, q in (("prefixed", prefixed), ("raw_question", unprefixed)):
                picks = (q[j] @ bank[j][:size].T).argsort(dim=-1, descending=True, stable=True)[:2].tolist()
                case[scenario][policy] = {"slots": picks, **evidence_metrics(picks, row["gold_sets"]),
                                          "outsiders": sum(i >= len(row["context"]) for i in picks)}
        values.append(case)
    return values


def visibility(reader, docs, truths, budget):
    if budget is not None:
        docs = [reader.tokenizer.decode(reader.tokenizer(t, max_length=budget, truncation=True,
                                                         add_special_tokens=False).input_ids) for t in docs]
    sequence = " " + " ".join(words(" ".join(docs))) + " "
    return any(f" {' '.join(words(answer))} " in sequence for answer in truths if words(answer))


def handle_reader(model_path, rows, replays):
    reader = Reader(model_path)
    outputs, requests = [], []
    for row, ref in zip(rows, replays, strict=True):
        sorted_sets = sorted(row["gold_sets"], key=lambda r: (len(r), r))
        support = sorted_sets[0]
        q = plain_question(row)
        actual = row["context"]
        chosen = ref["scenarios"]["clean"]["arms"]["context_pair"]["picked"]
        raw_gold = [actual[i][0] + ". " + " ".join(actual[i][1]) for i in support]
        controls = {
            "prefixed_gold_two": (row["question"], support[:2]),
            "raw_gold_two": (q, support[:2]),
            "raw_gold_all": (q, support),
            "raw_context_pair": (q, chosen),
            "raw_no_evidence": (q, []),
        }
        output = {"visibility_full_gold": visibility(reader, raw_gold, row["answers"], None),
                  "visibility_clipped_gold": visibility(reader, raw_gold, row["answers"], min(192, 384 // len(support))),
                  "arms": {}}
        for arm, (question, evidence) in controls.items():
            max_each = min(192, 384 // max(len(evidence), 1))
            chunks = [reader.tokenizer.decode(reader.tokenizer(actual[i][0] + ". " + " ".join(actual[i][1]),
                                                               max_length=max_each, truncation=True,
                                                               add_special_tokens=False).input_ids) for i in evidence]
            prompt = f"Answer the question with a short answer. Question: {question} Evidence: {' '.join(chunks)}"
            target = {}
            output["arms"][arm] = target
            requests.append((prompt, target, row["answers"]))
        outputs.append(output)
    unique = list(dict.fromkeys(p for p, _, _ in requests))
    responses = dict(zip(unique, reader.answers(unique), strict=True))
    for prompt, target, truths in requests:
        scores = [measure_answer(responses[prompt], answer) for answer in truths]
        target.update(em=max(x["em"] for x in scores), f1=max(x["f1"] for x in scores))
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "base", "fixture", "replay", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    data, replay = read(a.fixture), read(a.replay)
    assert replay["fixture_sha256"] == digest(a.fixture)
    rows = data["rows"][a.shard::a.shards]
    assert [r["id"] for r in rows] == [r["id"] for r in replay["rows"]]
    selection = select(rows, Encoder(a.assets / "encoder", 64))
    small = handle_reader(a.assets / "reader", rows, replay["rows"])
    base = handle_reader(a.base, rows, replay["rows"])
    result = [{"id": r["id"], "kind": r["kind"], "routing": route, "small": x, "base": y}
              for r, route, x, y in zip(rows, selection, small, base, strict=True)]
    save(a.output, {"fixture_sha256": digest(a.fixture), "replay_sha256": digest(a.replay), "device": torch.cuda.get_device_name(),
                    "base_files_sha256": digest(a.base / "files.sha256"), "rows": result, "elapsed_seconds": perf_counter() - start})


if __name__ == "__main__":
    main()
