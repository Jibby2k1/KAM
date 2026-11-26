#!/usr/bin/env python3
"""Independent GPU feasibility pilots for evidence use and episode routing."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from time import perf_counter
from zipfile import ZipFile

import torch
from torch.nn.functional import normalize
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from audit_kam_natural_data import digest, rank


def batches(items, size):
    return (items[i:i + size] for i in range(0, len(items), size))


class Encoder:
    def __init__(self, path, batch, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, use_safetensors=True).to(device).eval()
        self.batch = batch
        self.device = device

    @torch.inference_mode()
    def __call__(self, texts, prefix, max_length=192):
        unique = list(dict.fromkeys(texts))
        rows = []
        for chunk in batches(unique, self.batch):
            inputs = self.tokenizer([f"{prefix}: {text}" for text in chunk], padding=True,
                                    truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            out = self.model(**inputs).last_hidden_state
            rows.append(normalize((out * inputs.attention_mask.unsqueeze(-1)).sum(1), dim=-1))
        matrix = torch.cat(rows)
        vectors = dict(zip(unique, matrix))
        return torch.stack([vectors[text] for text in texts])


def normalized_answer(text):
    text = re.sub(r"[^\w\s]|\b(a|an|the)\b", " ", text.casefold())
    return text.split()


def measure_answer(prediction, target):
    p, t = normalized_answer(prediction), normalized_answer(target)
    if p == t:
        return {"em": 1, "f1": 1.0}
    common = sum((Counter(p) & Counter(t)).values())
    return {"em": 0, "f1": 2 * common / (len(p) + len(t))}


def candidate_texts(case):
    return [title + ". " + " ".join(sentences) for title, sentences in case["context"]]


def hotpot(args, encoder, audit):
    sample = {row["id"]: row for row in audit["sample"]}
    raw = {row["_id"]: row for row in json.loads(args.source.read_text()) if row["_id"] in sample}
    cases = [raw[id_] for id_ in sample]
    documents = [candidate_texts(case) for case in cases]
    flat = [text for pool in documents for text in pool]
    q = encoder([case["question"] for case in cases], "query")
    keys = encoder(flat, "passage").split([len(pool) for pool in documents])
    flat_scores = [pool @ query for query, pool in zip(q, keys)]
    selections = [scores.argsort(descending=True, stable=True).tolist() for scores in flat_scores]
    sequential = []
    second_call_seconds = None
    if args.sequential:
        torch.cuda.synchronize()
        second_started = perf_counter()
        augmented = [f"{case['question']} First retrieved document: {documents[i][ranked[0]]}"
                     for i, (case, ranked) in enumerate(zip(cases, selections))]
        for pool, original, query, ranked in zip(keys, flat_scores, encoder(augmented, "query"), selections):
            blended = (pool @ query + original) / 2
            blended[ranked[0]] = -torch.inf
            sequential.append([ranked[0], int(blended.argmax().item())])
        torch.cuda.synchronize()
        second_call_seconds = perf_counter() - second_started
    reader_path = args.assets / "reader"
    tokenizer = AutoTokenizer.from_pretrained(reader_path, local_files_only=True, use_fast=True)
    reader = AutoModelForSeq2SeqLM.from_pretrained(reader_path, local_files_only=True,
                                                    use_safetensors=True).cuda().eval()
    records, prompts = [], []
    for index, (case, ranked) in enumerate(zip(cases, selections)):
        audit_row = sample[case["_id"]]
        gold, bm25 = audit_row["gold"], audit_row["bm25"]
        texts = candidate_texts(case)
        gold_words = dict(case["context"])
        inputs = {
            "no_evidence": [],
            "gold_documents": [texts[x] for x in gold],
            "gold_sentences": [title + ". " + gold_words[title][index] for title, index in case["supporting_facts"]],
            "missing_one": [texts[gold[0]]],
            "bm25_top2": [texts[x] for x in bm25],
            "dense_top1": [texts[ranked[0]]],
            "dense_top2": [texts[x] for x in ranked[:2]],
        }
        routes = {"dense_top2": ranked[:2]}
        if args.sequential:
            routes["dense_sequential"] = sequential[index]
            inputs["dense_sequential"] = [texts[x] for x in sequential[index]]
        for arm, documents in inputs.items():
            evidence = " ".join(tokenizer.decode(tokenizer(doc, add_special_tokens=False, truncation=True,
                                                            max_length=192).input_ids) for doc in documents)
            prompts.append(f"Answer the question with a short answer. Question: {case['question']} Evidence: {evidence}")
            records.append({"id": case["_id"], "arm": arm, "dense_rank": routes.get(arm)})
    answers = []
    with torch.inference_mode():
        for chunk in batches(prompts, args.batch):
            encoded = tokenizer(chunk, padding=True, truncation=True, max_length=512,
                                return_tensors="pt").to("cuda")
            outputs = reader.generate(**encoded, max_new_tokens=20, do_sample=False)
            answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    for row, ans in zip(records, answers):
        row.update(measure_answer(ans, raw[row["id"]]["answer"]))
        if row["dense_rank"] is not None:
            row["both_support_docs"] = set(sample[row["id"]]["gold"]) <= set(row["dense_rank"][:2])
    if second_call_seconds is not None:
        print(json.dumps({"second_lookup_seconds": second_call_seconds, "queries": len(cases)}), flush=True)
    return records


def dialogue(args, encoder, audit):
    targets = {row["id"]: row for row in audit["sample"]}
    conversations = {id_.rsplit(":", 1)[0] for id_ in targets}
    with ZipFile(args.source) as archive:
        root = "multidoc2dial/multidoc2dial_"
        dialog = json.loads(archive.read(root + "dial_validation.json"))["dial_data"]
        documents = json.loads(archive.read(root + "doc.json"))["doc_data"]
    results = []
    for domain, cases in sorted(dialog.items()):
        chosen = [case for case in cases if case["dial_id"] in conversations]
        ids = sorted(documents[domain])
        texts = [documents[domain][id_]["title"] + ". " + documents[domain][id_]["doc_text"] for id_ in ids]
        document_matrix = encoder(texts, "passage")
        if not chosen:
            continue
        turns = [t for c in chosen for t in c["turns"]]
        queries = {t["utterance"]: v for t, v in zip(turns, encoder([t["utterance"] for t in turns], "query"))}
        passages = {t["utterance"]: v for t, v in zip(turns, encoder([t["utterance"] for t in turns], "passage"))}
        recent_vectors = {}
        if args.recent_window:
            recent_texts = {}
            for case in chosen:
                observed, pending = [], None
                for turn in case["turns"]:
                    if turn["role"] == "user":
                        pending = turn["utterance"]
                    elif pending is not None:
                        key = f"{case['dial_id']}:{turn['turn_id']}"
                        recent_texts[key] = pending + " Previous context: " + " ".join(observed[-2:])
                        observed.append(pending + " " + turn["utterance"])
            recent_vectors = dict(zip(recent_texts, encoder(list(recent_texts.values()), "query", max_length=256)))
        for case in chosen:
            pending, history = None, []
            for turn in case["turns"]:
                if turn["role"] == "user":
                    pending = turn["utterance"]
                    continue
                if pending is None:
                    continue
                key = f"{case['dial_id']}:{turn['turn_id']}"
                query = recent_vectors[key] if args.recent_window else queries[pending]
                base = (document_matrix @ query).cpu().tolist()
                index = max(range(len(base)), key=base.__getitem__)
                route = ids[index]

                def recall(eligible):
                    if not eligible:
                        return route
                    past_mat = torch.stack([h[1] for h in eligible])
                    memory_index = int((past_mat @ query).argmax().item())
                    recalled = eligible[memory_index][0]
                    if base[index] - base[ids.index(recalled)] <= .025:
                        return recalled
                    return route

                if key in targets:
                    gold, = {ref["doc_id"] for ref in turn["references"]}
                    results.append({"id": key, "domain": domain, "lag": targets[key]["eligible_turns_ago"],
                                    "static_dense": route == gold, "predicted_route_memory": recall(history) == gold,
                                    "last_two_predicted_route": recall(history[-2:]) == gold,
                                    "predicted_history_oracle": any(entry[0] == gold for entry in history),
                                    "static_lexical": ids[rank(pending, texts, 1)[0]] == gold})
                history.append((route, passages[turn["utterance"]]))
    if {row["id"] for row in results} != set(targets):
        raise RuntimeError("dialogue sample did not replay completely")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("track", choices=("hotpot", "dialogue"))
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--recent-window", action="store_true")
    p.add_argument("--sequential", action="store_true")
    a = p.parse_args()
    audit = json.loads(a.audit.read_text())
    assert audit["source_sha256"] == digest(a.source)
    started = perf_counter()
    enc = Encoder(a.assets / "encoder", a.batch)
    records = globals()[a.track](a, enc, audit)
    torch.cuda.synchronize()
    a.output.write_text(json.dumps({"status": "completed", "track": a.track, "split": "development",
                                    "source_sha256": audit["source_sha256"], "audit_sha256": digest(a.audit),
                                    "device": torch.cuda.get_device_name(), "torch": torch.__version__,
                                    "asset_revisions": json.loads((a.assets / "revisions.json").read_text()),
                                    "recent_window": a.recent_window,
                                    "sequential": a.sequential,
                                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "elapsed_seconds": perf_counter() - started, "records": records},
                                   indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
