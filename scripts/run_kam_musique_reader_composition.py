#!/usr/bin/env python3
"""Compare final questions with matched four-paragraph MuSiQue evidence and explicit bindings."""

import argparse
from functools import lru_cache
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT, binding_queries
from run_kam_musique_final_reader import PASSAGE_CAP
from run_kam_musique_recursive import REFERENCE
from run_kam_native_musique import corpus_and_fixture
from run_kam_natural_feasibility import batches, measure_answer, normalized_answer

SOURCE_ARMS = ("gold_paragraphs", "primary_selected")
PROMPTS = ("original_question", "last_unbound", "last_predicted", "last_rotated", "last_gold")
MAX_INPUT = 512
BODY_CAP = 72


def unpack(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def bind(template, values):
    return REFERENCE.sub(lambda matched: values[int(matched[1])-1], template).strip()


def tokenized_question(tokenizer, query):
    text = f"Answer the question with a short answer. Question: {query} Evidence:"
    return tokenizer(text, add_special_tokens=False).input_ids


def fixed_rows(tokenizer, prompts, passages):
    prefix = {name: tokenized_question(tokenizer, q) for name, q in prompts.items()}
    prompt_length = max(len(p) for p in prefix.values())
    cap = min(BODY_CAP, (MAX_INPUT-1-prompt_length)//len(passages))
    assert cap >= 16
    bodies = [tokenizer(text, add_special_tokens=False, truncation=True, max_length=cap).input_ids for text in passages]
    body = [i for v in bodies for i in v] + [tokenizer.eos_token_id]
    stamp = sha256(json.dumps(bodies, separators=(",", ":")).encode()).hexdigest()
    packed = {}
    for name, q in prefix.items():
        n = len(q)
        pad = prompt_length - n
        full = q + [tokenizer.pad_token_id]*pad + body
        mask = [1]*n + [0]*pad + [1]*len(body)
        assert len(full) == len(mask) <= MAX_INPUT
        packed[name] = tuple(full), tuple(mask)
    return packed, {"evidence_tokens": len(body)-1, "evidence_sha256": stamp, "total_length": prompt_length+len(body)}


@torch.inference_mode()
def read_packed(reader, pairs):
    records = list(dict.fromkeys(pairs))
    replies = {}
    for stack in batches(sorted(records, key=lambda p: len(p[0])), 16):
        limit = max(len(t) for t, _ in stack)
        inputs = torch.tensor([list(t) + [reader.tokenizer.pad_token_id] * (limit-len(t)) for t, _ in stack], device="cuda")
        masks = torch.tensor([list(m) + [0]*(limit-len(m)) for _, m in stack], device="cuda")
        ids = reader.model.generate(input_ids=inputs, attention_mask=masks, max_new_tokens=20, do_sample=False)
        replies.update(zip(stack, reader.tokenizer.batch_decode(ids, skip_special_tokens=True), strict=True))
    return [replies[p] for p in pairs]


def run(a, raw, fixture):
    fx, cx = digest(a.fixture), fixture["corpus_sha256"]
    labels = unpack(a.labels)
    templates, targets = unpack(a.templates), unpack(a.intermediate)
    gold_replay, primary = unpack(a.gold_reference), unpack(a.primary_reference)
    old = (templates, targets, labels, gold_replay, primary)
    assert all(r["fixture_sha256"] == fx for r in old)
    assert primary["corpus_sha256"] == gold_replay["corpus_sha256"] == cx
    assert primary["scoring_labels_sha256"] == digest(a.intermediate)
    assert primary["branch_labels_sha256"] == gold_replay["label_sha256"] == digest(a.labels)
    assert primary["templates_sha256"] == targets["templates_sha256"] == digest(a.templates)
    assert primary["models"]["google/flan-t5-base"] == gold_replay["reader_revision"] == BASE_REVISION
    assert all(r["shard"] == a.shard and r["shards"] == a.shards for r in (gold_replay, primary))
    path = a.assets/f"reader_base_{BASE_REVISION}"
    assert digest(path/"model.safetensors") == BASE_WEIGHT
    reader = Reader(path)

    @lru_cache(maxsize=4096)
    def passage(i, cap):
        row = raw[i]
        text = row["title"] + ". " + row["body"]
        ids = reader.tokenizer(text, add_special_tokens=False, truncation=True, max_length=cap).input_ids
        return reader.tokenizer.decode(ids)

    start = perf_counter()
    items = {group: [] for group in SOURCE_ARMS}
    for q, archive, actual in zip(fixture["rows"][a.shard::a.shards], gold_replay["rows"], primary["rows"], strict=True):
        id_ = q["id"]
        assert id_ == archive["id"] == actual["id"]
        if q["hops"] != 4:
            continue
        ordered = targets["rows"][id_]["targets"]
        sorted_gold = archive["arms"]["gold_only"]["evidence_ids"]
        assert len(ordered) == 4 and sorted(ordered) == sorted_gold == q["gold"]
        selected = actual["arms"]["top1"]["fixed_evidence_ids"]
        assert len(selected) == len(set(selected)) == 4
        for group, ids, aligned, em in (
            ("gold_paragraphs", sorted_gold, ordered, archive["arms"]["gold_only"]["em"]),
            ("primary_selected", selected, selected, actual["arms"]["top1"]["fixed_answer_em"]),
        ):
            items[group].append({"q": q, "context": ids, "aligned": aligned, "archived_exact": em,
                                 "templates": templates["templates"][id_], "truth": targets["rows"][id_]["answers"],
                                 "accepted": labels["answers"][id_], "predicted": []})

    results = {}
    for group, tested in items.items():
        prompt0 = [f"Answer the question with a short answer. Question: {v['q']['question']} Evidence: "
                   + " ".join(passage(i, PASSAGE_CAP) for i in v["context"]) for v in tested]
        for v, answer in zip(tested, reader.answers(prompt0), strict=True):
            matched = max(measure_answer(answer, label)["em"] for label in v["accepted"])
            assert bool(matched) == bool(v["archived_exact"])
        for step in range(3):
            resolved = binding_queries([v["templates"] for v in tested], [v["predicted"] for v in tested], step)
            prompts = [f"Answer the question with a short answer. Question: {query} Evidence: {passage(v['aligned'][step],128)}"
                       for v, query in zip(tested, resolved, strict=True)]
            for v, txt in zip(tested, reader.answers(prompts), strict=True):
                v["predicted"].append(" ".join(txt.split()) or "unknown")
        outputs, prompts = [], []
        for v in tested:
            path, predict, labels = v["templates"], v["predicted"], v["truth"]
            assert max(measure_answer(labels[-1], x)["em"] for x in v["accepted"]) == 1
            choices = {
                "original_question": v["q"]["question"],
                "last_unbound": bind(path[-1], ["unknown"]*3),
                "last_predicted": bind(path[-1], predict),
                "last_rotated": bind(path[-1], predict[1:]+predict[:1]),
                "last_gold": bind(path[-1], labels[:3]),
            }
            record, audit = fixed_rows(reader.tokenizer, choices, [passage(i, PASSAGE_CAP) for i in v["context"]])
            prompts.extend(record.values())
            direct_refs = set(map(lambda m: int(m)-1, REFERENCE.findall(path[-1])))
            final_terms = {tuple(normalized_answer(s)) for s in v["accepted"]}
            gold_overlap = any(tuple(normalized_answer(labels[i])) in final_terms for i in direct_refs)
            outputs.append({"id": v["q"]["id"], "evidence": v["context"], "archived_exact": bool(v["archived_exact"]),
                            "prediction_exact": [bool(measure_answer(x,y)["em"]) for x,y in zip(predict, labels[:3], strict=True)],
                            "gold_binding_matches_final": gold_overlap,
                            "rotation_active": choices["last_rotated"] != choices["last_predicted"], "audit": audit})
        decoded = read_packed(reader, prompts)
        for j, (v, row) in enumerate(zip(tested, outputs, strict=True)):
            row["answers"] = {arm: bool(max(measure_answer(out, y)["em"] for y in v["accepted"]))
                              for arm, out in zip(PROMPTS, decoded[j*len(PROMPTS):(j+1)*len(PROMPTS)], strict=True)}
        results[group] = outputs
        print(group, len(outputs), len(set(prompts)), flush=True)
    return {"fixture_sha256": fx, "corpus_sha256": cx, "reader_revision": BASE_REVISION,
            "prior": {"gold": digest(a.gold_reference), "primary": digest(a.primary_reference)},
            "shard": a.shard, "shards": a.shards, "device": torch.cuda.get_device_name(),
            "max_tokens": MAX_INPUT, "max_body_cap": BODY_CAP, "elapsed_seconds": perf_counter()-start, "panels": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("fixture", "corpus", "assets", "labels", "templates", "intermediate", "gold-reference", "primary-reference", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    raw, fixture = corpus_and_fixture(a)
    torch.set_float32_matmul_precision("highest")
    with torch.inference_mode():
        result = run(a, raw, fixture)
    a.output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
