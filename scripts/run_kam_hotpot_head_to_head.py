#!/usr/bin/env python3
"""Same-split, paired learned single-shot and conditioned sequential retrieval."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn.functional import log_softmax
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from audit_kam_natural_data import digest
from run_kam_natural_feasibility import Encoder, batches, candidate_texts, measure_answer
from run_kam_natural_router import QueryAdapter, score, to_device


def encode(args, cases):
    ospec = torch.load(args.training_cache, map_location="cpu", weights_only=True)
    manifest = json.loads(args.partition.read_text())
    assert manifest["training_partition_sha256"] == ospec["partition_sha256"]
    enc = Encoder(args.assets / "encoder", 64)
    documents = [doc for case in cases for doc in candidate_texts(case)]
    q = enc([case["question"] for case in cases], "query").cpu()
    k = enc(documents, "passage").reshape(len(cases), 10, -1).cpu()
    torch.save({"queries": q, "keys": k, "ids": manifest["ids"], "partition_sha256": digest(args.partition)},
               args.fresh_cache)


def adapter(args, spec, device="cuda"):
    cache = torch.load(args.training_cache, map_location="cpu", weights_only=True)
    q, k, gold = to_device(cache["training"], device)
    seed = spec["seeds"][args.shard]
    torch.manual_seed(seed)
    model = QueryAdapter(q.shape[-1], spec["bottleneck"], False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], fused=q.is_cuda)
    rng = torch.Generator(device=device).manual_seed(seed + 40_000)
    for _ in range(spec["epochs"]):
        for indices in torch.randperm(len(q), generator=rng, device=device).split(spec["batch"]):
            values = score(model(q[indices]), k[indices]) * spec["score_scale"]
            loss = -(log_softmax(values, dim=-1) * gold[indices]).sum(-1).mean() / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite trained query adapter")
    return model.eval()


@torch.inference_mode()
def route(enc, model, cases, queries, keys, *, include_frozen=True):
    selected, timing = {}, {}
    policies = [("learned", model)]
    if include_frozen:
        policies.insert(0, ("frozen", lambda q: q))
    for policy, transform in policies:
        if queries.is_cuda:
            torch.cuda.synchronize()
        start = perf_counter()
        first_scores = score(transform(queries), keys)
        shot = first_scores.argsort(descending=True, stable=True)[:, :2].tolist()
        timing[policy + "_single_shot_rank_seconds"] = perf_counter() - start
        start = perf_counter()
        augmented = [
            f"{case['question']} First retrieved document: {candidate_texts(case)[top[0]]}"
            for case, top in zip(cases, shot)
        ]
        second_queries = enc(augmented, "query")
        blended = (first_scores + score(transform(second_queries), keys)) * .5
        first = torch.tensor([top[0] for top in shot], device=queries.device).unsqueeze(-1)
        blended.scatter_(1, first, -torch.inf)
        second = torch.cat((first, blended.argmax(-1, keepdim=True)), dim=1).tolist()
        timing[policy + "_additional_retrieval_seconds"] = perf_counter() - start
        selected[policy + "_single_shot"] = shot
        selected[policy + "_sequential"] = second
    return selected, timing


def reader(args, cases, selected):
    reader_dir = args.assets / "reader"
    tokenizer = AutoTokenizer.from_pretrained(reader_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(reader_dir, local_files_only=True,
                                                  use_safetensors=True).cuda().eval()
    rows, prompts = [], []
    for arm, selections in selected.items():
        for case, chosen in zip(cases, selections):
            candidates = candidate_texts(case)
            gold = {title for title, _ in case["supporting_facts"]}
            evidence = " ".join(tokenizer.decode(tokenizer(candidates[i], add_special_tokens=False,
                                      max_length=192, truncation=True).input_ids) for i in chosen)
            prompts.append(f"Answer the question with a short answer. Question: {case['question']} Evidence: {evidence}")
            rows.append({"id": case["_id"], "arm": arm, "selected": chosen,
                         "both_support": gold <= {case["context"][i][0] for i in chosen}})
    unique = list(dict.fromkeys(prompts))
    decoded = []
    with torch.inference_mode():
        for chunk in batches(unique, 32):
            data = tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            out = model.generate(**data, max_new_tokens=20, do_sample=False)
            decoded.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    answers = {case["_id"]: case["answer"] for case in cases}
    predictions = dict(zip(unique, decoded))
    for row, prompt in zip(rows, prompts):
        row.update(measure_answer(predictions[prompt], answers[row["id"]]))
    return rows, len(unique)


def run(args, cases):
    spec = json.loads(args.config.read_text())
    cache = torch.load(args.fresh_cache, map_location="cpu", weights_only=True)
    assert cache["partition_sha256"] == digest(args.partition) and cache["ids"] == [c["_id"] for c in cases]
    started = perf_counter()
    model = adapter(args, spec)
    enc = Encoder(args.assets / "encoder", 64)
    selected, timing = route(enc, model, cases, cache["queries"].cuda(), cache["keys"].cuda())
    del enc
    gold_rows = []
    for case in cases:
        gold = {title for title, _ in case["supporting_facts"]}
        gold_rows.append([i for i, (title, _) in enumerate(case["context"]) if title in gold])
    selected["gold"] = gold_rows
    records, unique = reader(args, cases, selected)
    args.output.write_text(json.dumps({
        "status": "completed", "scope": "fresh public development only",
        "seed": spec["seeds"][args.shard], "partition_sha256": digest(args.partition),
        "config_sha256": digest(args.config), "torch": torch.__version__, "device": torch.cuda.get_device_name(),
        "models": json.loads((args.assets / "revisions.json").read_text()),
        "timing": timing, "elapsed_seconds": perf_counter() - started, "unique_reader_prompts": unique,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "records": records,
    }, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("encode", "run"))
    for arg in ("source", "partition", "assets", "training-cache", "fresh-cache", "config"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--shard", type=int)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    spec = json.loads(a.partition.read_text())
    assert spec["source_sha256"] == digest(a.source)
    subset = set(spec["ids"])
    source = {case["_id"]: case for case in json.loads(a.source.read_text()) if case["_id"] in subset}
    cases = [source[id_] for id_ in spec["ids"]]
    (encode if a.mode == "encode" else run)(a, cases)


if __name__ == "__main__":
    main()
