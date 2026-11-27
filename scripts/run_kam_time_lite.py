#!/usr/bin/env python3
"""Measure choice recovery from fixed TIME-Lite contexts and date interventions."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from kam_flan_reader_assets import LARGE_REVISION, LARGE_WEIGHT
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT
from run_kam_natural_feasibility import Encoder, batches

ARMS = ("closed_book", "prefix", "dense", "dense_date_masked", "dense_chronological", "dense_dates_rotated")
WIDTH = 1024
WINDOWS = 4


def token_ids(tokenizer, value):
    return tokenizer(value, add_special_tokens=False, verbose=False).input_ids


def stem(q, rotation):
    options = q["options"]
    labels = list("ABCD"[:len(options)])
    arranged = options[rotation:] + options[:rotation]
    formatted = "\n".join(f"{tag}. {v}" for tag, v in zip(labels, arranged, strict=True))
    return f"Read the context and answer with the letter of the correct option.\nQuestion: {q['question']}\n{formatted}\n"


def attach(prefix, cue, window_labels, bodies, order, rotated, arm, maximum, pad, eos):
    assert len(window_labels) == len(bodies)
    fixed_headers = sum(window_labels[i][0] for i in range(len(window_labels)))
    remaining = maximum - len(prefix) - len(cue) - 1 - fixed_headers
    assert remaining >= 0
    n = len(order)
    lengths = [min(remaining // max(n, 1), len(body)) for body in bodies]
    left = remaining - sum(lengths)
    for i in range(n):
        use = min(left, len(bodies[i]) - lengths[i])
        lengths[i] += use
        left -= use
    body_sha = sha256(json.dumps([b[:length] for b, length in zip(bodies, lengths, strict=True)],
                                 separators=(",", ":")).encode()).hexdigest()
    tokens, mask = prefix.copy(), [1] * len(prefix)
    for i in order:
        slot, date_tokens = window_labels[i]
        if arm == "dense_dates_rotated":
            date_tokens = window_labels[rotated[i]][1]
        if arm == "dense_date_masked":
            date_tokens = []
        assert len(date_tokens) <= slot
        body = bodies[i][:lengths[i]]
        tokens += date_tokens + [pad] * (slot - len(date_tokens)) + body
        mask += [1] * len(date_tokens) + [0] * (slot - len(date_tokens)) + [1] * len(body)
    tokens += cue + [eos]
    mask += [1] * (len(cue) + 1)
    assert len(tokens) == len(mask) <= maximum
    return tokens, mask, body_sha, sum(lengths)


def prompts_for(tokenizer, cases, contexts, ranked, arms):
    output = {a: [] for a in arms}
    reports = []
    headers = [[token_ids(tokenizer, f"Date: {w['day']}.") if w["day"] else [] for w in ctx["windows"]] for ctx in contexts]
    bodies = [tokenizer([w["body"] for w in ctx["windows"]], add_special_tokens=False, verbose=False).input_ids
              for ctx in contexts]
    date_slot = max((len(label) for context_labels in headers for label in context_labels), default=0)
    instruction = token_ids(tokenizer, "Answer:")
    for q, dense in zip(cases, ranked, strict=True):
        j = q["context"]
        this = contexts[j]["windows"]
        # All dense variants retain the exact same text and token quotas.
        prefix_ids = list(range(min(WINDOWS, len(this))))
        report = {"id": q["id"], "dense": dense, "prompt": {}}
        rotations = len(q["options"])
        for arm in arms:
            selected = [] if arm == "closed_book" else (prefix_ids if arm == "prefix" else dense)
            order = list(range(len(selected)))
            dates = [this[i]["day"] for i in selected]
            if arm == "dense_chronological":
                order.sort(key=lambda i: (dates[i], selected[i]))
            rotate = order[1:] + order[:1]
            labels = [(date_slot, headers[j][idx]) for idx in selected]
            chosen = [bodies[j][idx] for idx in selected]
            audits = []
            for r in range(rotations):
                prefix = token_ids(tokenizer, stem(q, r) + ("Context:\n" if selected else ""))
                ids, mask, body_hash, n = attach(prefix, instruction, labels, chosen, order, rotate, arm,
                                                WIDTH, tokenizer.pad_token_id, tokenizer.eos_token_id)
                output[arm].append((ids, mask, rotations, r))
                audits.append({"body_sha256": body_hash, "body_tokens": n, "total_length": len(ids)})
            report["prompt"][arm] = {"selected": selected, "order": order,
                                     "rotated_date_changes": (sum(dates[k] != dates[rotate[k]] for k in order)
                                                              if arm == "dense_dates_rotated" else 0),
                                     "rotations": audits}
        reports.append(report)
    return output, reports, date_slot


def evaluate(model, items, batch):
    tokenizer = model.tokenizer
    label_ids = [token_ids(tokenizer, c) for c in "ABCD"]
    assert all(len(x) == 1 for x in label_ids)
    label_tensor = torch.tensor([v[0] for v in label_ids], device="cuda")
    assert label_tensor.unique().numel() == 4
    restored = [None] * len(items)
    sequence = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    for block in batches(sequence, batch):
        selected = [items[i] for i in block]
        width = max(len(t) for t, _, _, _ in selected)
        tokens = torch.tensor([t + [tokenizer.pad_token_id] * (width-len(t)) for t, _, _, _ in selected], device="cuda")
        attention = torch.tensor([mask + [0] * (width-len(mask)) for _, mask, _, _ in selected], device="cuda")
        initial = torch.full((len(block), 1), model.model.config.decoder_start_token_id,
                             device="cuda", dtype=torch.long)
        logits = model.model(input_ids=tokens, attention_mask=attention, decoder_input_ids=initial,
                             use_cache=False).logits[:, 0].index_select(1, label_tensor).float().tolist()
        for idx, row, (_, _, count, rotation) in zip(block, logits, selected, strict=True):
            shifted = row[:count]
            restored[idx] = shifted[-rotation:] + shifted[:-rotation] if rotation else shifted
    return restored


@torch.inference_mode()
def run(a):
    packed = json.loads(gzip.decompress(a.fixture.read_bytes()))
    cases, contexts = (packed[k] for k in ("questions", "contexts"))
    n = len(cases)
    by_context = [[w["body"] for w in c["windows"]] for c in contexts]
    marks = [0]
    for items in by_context:
        marks.append(marks[-1] + len(items))
    encoder = Encoder(a.assets / "encoder", 128)
    torch.cuda.synchronize()
    start = perf_counter()
    passages = encoder([t for row in by_context for t in row], "passage")
    queries = encoder([q["question"] for q in cases], "query")
    ranks = []
    for i, q in enumerate(cases):
        j = q["context"]
        vectors = passages[marks[j]:marks[j+1]]
        scores = vectors @ queries[i]
        ranks.append(scores.argsort(descending=True, stable=True)[:WINDOWS].tolist())
    torch.cuda.synchronize()
    retrieval_s = perf_counter() - start
    del encoder, passages, queries
    revision, checksum = (BASE_REVISION, BASE_WEIGHT) if a.size == "base" else (LARGE_REVISION, LARGE_WEIGHT)
    folder = a.assets / f"reader_{a.size}_{revision}"
    assert digest(folder / "model.safetensors") == checksum
    model = Reader(folder)
    arms = ARMS if a.domain != "TIME-Lite-Wiki" else ("closed_book", "prefix", "dense", "dense_chronological")
    data, returned, date_slot = prompts_for(model.tokenizer, cases, contexts, ranks, arms)
    evaluate(model, data["dense"][:a.batch], a.batch)
    timings = {}
    for arm in Random(13965).sample(list(arms), len(arms)):
        torch.cuda.synchronize()
        start = perf_counter()
        scores = evaluate(model, data[arm], a.batch)
        torch.cuda.synchronize()
        timings[arm] = perf_counter() - start
        i = 0
        for q, row in zip(cases, returned, strict=True):
            m = len(q["options"])
            row.setdefault("logits", {})[arm] = scores[i:i+m]
            i += m
        assert i == len(scores)
        print(a.domain, a.size, arm, n, timings[arm], flush=True)
    return {"domain": a.domain, "size": a.size, "fixture_sha256": digest(a.fixture), "encoder_revisions":
            json.loads((a.assets/"revisions.json").read_bytes()), "model": {"revision": revision, "sha256": checksum},
            "device": torch.cuda.get_device_name(), "input_tokens_max": WIDTH, "views": WINDOWS,
            "batch": a.batch, "date_slots_per_view": date_slot, "encode_rank_seconds": retrieval_s, "reader_seconds": timings,
            "rows": returned}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("assets", "fixture", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--size", choices=("base", "large"), required=True)
    p.add_argument("--domain", choices=("TIME-Lite-Wiki", "TIME-Lite-News", "TIME-Lite-Dial"), required=True)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    data = run(a)
    a.output.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
