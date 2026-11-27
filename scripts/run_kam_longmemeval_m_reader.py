#!/usr/bin/env python3
"""Read frozen LongMemEval turn sets with and without conversation dates."""

import argparse
from collections import deque
from datetime import datetime
import gzip
import json
from pathlib import Path
from time import perf_counter

from audit_kam_natural_data import digest

MAX_TURNS = 20
MAX_INPUT = 512
ARMS = ("dense", "dense_dated", "user", "marked_in_dense", "oracle_marked", "oracle_marked_dated", "no_history")


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def prepare_labels(a):
    from prepare_kam_longmemeval_m import SOURCE_HASH, records

    assert digest(a.source) == SOURCE_HASH
    fixture = read(a.fixture)
    answers = {}
    for raw, q in zip(records(a.source), fixture["rows"], strict=True):
        assert raw["question_id"] == q["id"] and raw["question"] == q["question"]
        assert type(raw["answer"]) in (str, int)
        answers[q["id"]] = str(raw["answer"])
    a.output.write_text(json.dumps({"fixture_sha256": digest(a.fixture), "answers": answers},
                                   ensure_ascii=False, separators=(",", ":")) + "\n")


def marked(item, sessions):
    gold = set(item["gold_ids"])
    per_session = {}
    for slot in sessions:
        sid = item["source_session_ids"][slot]
        if sid in gold and item["author_turn_ids"][slot] and sid not in per_session:
            per_session[sid] = deque((slot, pos) for pos in item["author_turn_ids"][slot])
    result = []
    while any(per_session.values()) and len(result) < MAX_TURNS:
        for turns in per_session.values():
            if turns:
                result.append(turns.popleft())
                if len(result) == MAX_TURNS:
                    break
    return result


def selection(q, baseline):
    dense = [tuple(p) for p in baseline["turns_from_top5"]["dense"]]
    user = [tuple(p) for p in baseline["turns_from_top5"]["user"]]
    marked_dense = marked(q, baseline["full"]["dense"][:5])
    mixed = list(dict.fromkeys(marked_dense + dense))[:MAX_TURNS]
    oracle = marked(q, range(len(q["slots"])))
    return dict(dense=dense, dense_dated=dense, user=user, marked_in_dense=mixed, oracle_marked=oracle,
                oracle_marked_dated=oracle, no_history=[])


def date(minutes):
    d = datetime.fromordinal(minutes // 1440)
    return d.strftime("%Y-%m-%d")


def token_prompt(tokenizer, item, positions, bank, dated):
    def ids(text):
        return tokenizer(text, add_special_tokens=False).input_ids

    question = ids("Answer the question with a short answer. If the history does not give an answer, say unknown.")
    question += ids("Question: " + item["question"])
    clock = ids("Question date: " + date(item["time"]))
    labels, details = [], []
    for slot, turn in positions:
        role, content = bank[item["slots"][slot]][turn]
        labels.append((ids(role + ":"), ids(date(item["dates"][slot]) + " " + role + ":")))
        details.append(ids(content))
    available = MAX_INPUT - 1 - len(question) - len(clock) - len(ids("History:"))
    available -= sum(max(map(len, pair)) for pair in labels)
    assert available >= 0
    quotas = [0] * len(details)
    active = [i for i, tokens in enumerate(details) if tokens]
    while active and available:
        allotment = max(1, available // len(active))
        for i in active:
            gained = min(len(details[i]) - quotas[i], allotment, available)
            quotas[i] += gained
            available -= gained
        active = [i for i in active if quotas[i] < len(details[i])]
    tokens = question + (clock if dated else []) + ids("History:")
    for label, body, length in zip(labels, details, quotas, strict=True):
        tokens += label[int(dated)] + body[:length]
    tokens.append(tokenizer.eos_token_id)
    assert len(tokens) <= MAX_INPUT
    return tokens, [sum(quotas), sum(map(len, details)), sum(n < len(s) for n, s in zip(quotas, details, strict=True))]


def run(a):
    import torch

    from run_kam_hotpot_pool_scaling import Reader
    from run_kam_longmemeval_m import iter_bank
    from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT
    from run_kam_natural_feasibility import batches, measure_answer, normalized_answer

    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    record = read(a.reference)
    fixture = read(a.fixture)
    index = read(a.bank_index)
    fx = digest(a.fixture)
    assert fx == record["fixture_sha256"] == index["fixture_sha256"]
    assert index["bank_sha256"] == record["bank_sha256"] == fixture["session_bank_sha256"]
    assert (record["shard"], record["shards"]) == (a.shard, a.shards)
    cases = fixture["rows"][a.shard::a.shards]
    planned = []
    need = set()
    for q, retrieved in zip(cases, record["rows"], strict=True):
        assert q["id"] == retrieved["id"]
        groups = selection(q, retrieved)
        planned.append((q, groups))
        need.update(q["slots"][slot] for positions in groups.values() for slot, _ in positions)
    bank = {}
    for j, path in enumerate(a.banks):
        assert digest(path) == index["shards"][j]
        for item in iter_bank(path):
            if item["id"] in need:
                bank[item["id"]] = item["turns"]
    assert set(bank) == need
    model_path = a.assets / f"reader_base_{BASE_REVISION}"
    assert digest(model_path / "model.safetensors") == BASE_WEIGHT
    reader = Reader(model_path)
    prompts, panels = {}, []
    for q, groups in planned:
        panel = {"id": q["id"], "arms": {}}
        for arm, positions in groups.items():
            tokens, content = token_prompt(reader.tokenizer, q, positions, bank, arm.endswith("_dated"))
            key = tuple(tokens)
            prompts.setdefault(key, [])
            marked_ids = {q["source_session_ids"][slot] for slot, turn in positions
                          if turn in q["author_turn_ids"][slot]}
            got = {"retained_turn_tokens": content[0], "original_turn_tokens": content[1],
                   "clipped_turns": content[2], "prompt_tokens": len(tokens),
                   "marked_support": set(q["gold_ids"]) <= marked_ids,
                   "turns": [list(pair) for pair in positions]}
            panel["arms"][arm] = got
            prompts[key].append((q["id"], got))
        for arm in ("dense", "oracle_marked"):
            assert panel["arms"][arm]["retained_turn_tokens"] == panel["arms"][arm + "_dated"]["retained_turn_tokens"]
        panels.append(panel)

    answers = read(a.labels)
    assert answers["fixture_sha256"] == fx and set(answers["answers"]) == {q["id"] for q in fixture["rows"]}
    sorted_input = sorted(prompts, key=len)
    with torch.inference_mode():
        for group in batches(sorted_input, 16):
            inputs = reader.tokenizer.pad({"input_ids": [list(tokens) for tokens in group]},
                                          padding=True, return_tensors="pt").to("cuda")
            output = reader.model.generate(**inputs, max_new_tokens=20, do_sample=False)
            predicted = reader.tokenizer.batch_decode(output, skip_special_tokens=True)
            for tokens, answer in zip(group, predicted, strict=True):
                norm = normalized_answer(answer)
                abstain = norm in (["unknown"], ["i", "don", "t", "know"], ["cannot", "answer"])
                for qid, pointer in prompts[tokens]:
                    pointer.update(measure_answer(answer, answers["answers"][qid]), abstained=abstain)
    payload = {"fixture_sha256": fx, "reference_sha256": digest(a.reference), "labels_sha256": digest(a.labels),
               "reader_revision": BASE_REVISION, "reader_weights_sha256": BASE_WEIGHT, "device": torch.cuda.get_device_name(),
               "shard": a.shard, "shards": a.shards, "unique_prompts": len(prompts), "arms": ARMS,
               "elapsed_seconds": perf_counter() - start, "rows": panels}
    a.output.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("labels", "read"))
    for name in ("fixture", "output", "source", "labels", "reference", "assets", "bank-index"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--banks", nargs=8, type=Path)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    (prepare_labels if a.mode == "labels" else run)(a)


if __name__ == "__main__":
    main()
