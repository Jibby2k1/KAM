#!/usr/bin/env python3
"""Compare BM25 top one with model reranking of BM25 top two on native pools."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from audit_kam_natural_data import rank
from kam_adapter_preservation import digest, key
from kam_math_kl_protocol import test_runs, write_json

PROTOCOL = "kam-musique-native-rerank-v1"
ROUTES = ("bm25_top1", "bm25_rerank2")


def evaluate(root, work, index):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import CONTEXT, next_logits, pad

    frozen, fixture, selection = (work / name for name in ("manifest.json", "panel.json", "selection.json"))
    manifest, data, chosen = [json.loads(p.read_text()) for p in (frozen, fixture, selection)]
    assert manifest["protocol"] == PROTOCOL and tuple(manifest["routes"]) == ROUTES
    assert manifest["panel_sha256"] == digest(fixture)
    assert (chosen["selected_recipe"], chosen["selected_kl_weight"]) == ("lower_lr_kl05", .5)
    run = test_runs(chosen["selected_recipe"])[index]
    model, tokenizer, stack, metadata = load_checkpoint(root, *run)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    letters = [tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
    assert all(len(x) == 1 for x in letters) and letters[0] != letters[1]
    labels = [x[0] for x in letters]

    def clipped(text):
        ids = tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)
        return tokenizer.decode(ids, skip_special_tokens=True)

    def choice(message):
        text = tokenizer.apply_chat_template([{"role": "user", "content": message}],
                                             tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) <= CONTEXT
        x, mask = pad([ids], tokenizer.pad_token_id)
        logits, _ = next_logits(model, stack, x, mask)
        a, b = logits[0, labels].float().tolist()
        return b - a, len(ids)

    def route(method, texts, clipped_pool, earlier_order, endpoints, question):
        ranked = rank(question, [texts[i] for i in earlier_order], limit=2)
        first, other = (earlier_order[i] for i in ranked)
        tokens = 0
        if method == "bm25_rerank2":
            margin, tokens = choice("Select the passage that contains earlier evidence needed to answer the question. "
                                    "Reply with A or B.\n\nQuestion: " + question +
                                    "\n\nPassage A:\n" + clipped_pool[first] +
                                    "\n\nPassage B:\n" + clipped_pool[other] + "\n\nChoice:")
            if margin > 0:
                first = other
        margin, endpoint_tokens = choice("Select the passage that answers the question. "
                                         "Use earlier evidence to identify the right entity when it is available. "
                                         "Reply with A or B.\n\nQuestion: " + question +
                                         "\n\nEarlier passage:\n" + clipped_pool[first] +
                                         "\n\nPassage A:\n" + endpoints[0] +
                                         "\n\nPassage B:\n" + endpoints[1] + "\n\nChoice:")
        return margin, tokens + endpoint_tokens, first

    records = []
    for pair in data["pairs"]:
        endpoints = [clipped(q["evidence"][-1]["title"] + ". " + q["evidence"][-1]["body"]) for q in pair["questions"]]
        for index, item in enumerate(pair["questions"]):
            texts = data["first_hop_pools"][item["id"]]
            clipped_pool = [clipped(text) for text in texts]
            orders = [list(range(len(texts))), list(reversed(range(len(texts))))]
            if not records:
                for method in ROUTES:
                    route(method, texts, clipped_pool, orders[0], endpoints, item["question"])
            for reverse in (False, True):
                offered = endpoints[::-1] if reverse else endpoints
                expected = index ^ reverse
                for slot, order in enumerate(orders):
                    record = {"pair": pair["id"], "family": pair["family"], "question": item["id"],
                              "reverse": reverse, "expected": expected, "earlier_slot": slot}
                    methods = ROUTES if (index ^ reverse ^ slot) == 0 else ROUTES[::-1]
                    for method in methods:
                        observations, elapsed = [], []
                        for _ in range(3):
                            torch.cuda.synchronize()
                            start = perf_counter_ns()
                            observations.append(route(method, texts, clipped_pool, order, offered, item["question"]))
                            torch.cuda.synchronize()
                            elapsed.append((perf_counter_ns() - start) / 1e6)
                        assert all(result == observations[0] for result in observations)
                        margin, tokens, first = observations[0]
                        record[method] = {"correct": margin != 0 and int(margin > 0) == expected,
                                          "first_correct": first == index, "margin": margin,
                                          "tokens": tokens, "median_ms": median(elapsed)}
                    records.append(record)
    write_json(work / f"routes-{key(*run)}.json", {"metadata": {**metadata, "protocol": PROTOCOL, "gpu": gpu,
               "manifest_sha256": digest(frozen), "selection_sha256": digest(selection),
               "excluded_pairs": manifest["excluded_pairs"]}, "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate",))
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    evaluate(args.root, args.work, args.index)
