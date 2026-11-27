#!/usr/bin/env python3
"""Fit a MuSiQue subquestion generator using only disjoint training questions."""

import argparse
import gzip
import json
from pathlib import Path
from random import Random
import re
from time import perf_counter

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from audit_kam_natural_data import digest, words

REFERENCE = re.compile(r"#(\d+)\b")
SEPARATOR = "<extra_id_0>"
HOPS = (2, 3, 4)
TOKEN_CAP = 160
MODEL_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"


def instruction(q):
    return f"Break the question into short ordered subquestions. Refer to earlier answers with #1, #2, and #3. Question: {q}"


def normalized_tmpl(text):
    return " ".join(words(text))


def interfaces(assets):
    model_config = json.loads((assets / "revisions.json").read_text())
    assert model_config["google/flan-t5-small"] == MODEL_REVISION
    reader = assets / "reader"
    tokenizer = AutoTokenizer.from_pretrained(reader, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(reader, local_files_only=True, use_safetensors=True).cuda()
    return tokenizer, model


def structure(tokens, tokenizer):
    marker = tokenizer.convert_tokens_to_ids(SEPARATOR)
    end = tokenizer.eos_token_id
    items = [[]]
    finished = False
    for token in tokens:
        if token == end:
            finished = True
            break
        if token == marker:
            items.append([])
        else:
            items[-1].append(token)
    steps = [" ".join(tokenizer.decode(r, skip_special_tokens=True).split()) for r in items]
    valid = finished and len(steps) in HOPS
    valid &= all(part and all(1 <= int(i) <= n for i in REFERENCE.findall(part))
                 for n, part in enumerate(steps))
    valid &= any(REFERENCE.search(p) for p in steps)
    return steps, bool(valid)


@torch.inference_mode()
def predict(model, tokenizer, questions, batch=32):
    model.eval()
    outputs = []
    for start in range(0, len(questions), batch):
        text = list(map(instruction, questions[start:start+batch]))
        value = tokenizer(text, padding=True, truncation=True, max_length=TOKEN_CAP, return_tensors="pt").to("cuda")
        sample = model.generate(**value, max_new_tokens=TOKEN_CAP, do_sample=False)
        outputs.extend(structure(t, tokenizer) for t in sample.tolist())
    return outputs


def summarize(model, tokenizer, rows):
    predicted = predict(model, tokenizer, [q["question"] for q in rows])
    metrics = {h: {"cases": 0, "valid": 0, "same_length": 0, "same_references": 0, "exact": 0} for h in HOPS}
    for sample, (steps, valid) in zip(rows, predicted, strict=True):
        gold = sample["steps"]
        result = metrics[len(gold)]
        result["cases"] += 1
        result["valid"] += valid
        result["same_length"] += len(steps) == len(gold)
        result["same_references"] += [REFERENCE.findall(x) for x in steps] == [REFERENCE.findall(x) for x in gold]
        result["exact"] += list(map(normalized_tmpl, steps)) == list(map(normalized_tmpl, gold))
    return metrics


def fit(a):
    start = perf_counter()
    digest_data = digest(a.data)
    panel = json.loads(gzip.decompress(a.data.read_bytes()))
    audit = json.loads(gzip.decompress(a.manifest.read_bytes()))
    assert digest_data == audit["data_sha256"] and panel["fixture_sha256"] == audit["fixture_sha256"]
    train, valid = (panel["splits"][name] for name in ("fit", "validation"))
    assert [q["id"] for q in train] == [q["id"] for q in audit["ids"]["fit"]]
    assert [q["id"] for q in valid] == [q["id"] for q in audit["ids"]["validation"]]
    title_keys = lambda rows: {t for row in rows for t in row["support_title_sha256"]}
    assert title_keys(train).isdisjoint(title_keys(valid))
    tokenizer, model = interfaces(a.assets)
    vocab = tokenizer.pad_token_id
    marker = tokenizer.convert_tokens_to_ids(SEPARATOR)
    queries = tokenizer([instruction(q["question"]) for q in train], padding="max_length",
                        max_length=TOKEN_CAP, truncation=True, return_tensors="pt").to("cuda")
    tokens = tokenizer([" " + f" {SEPARATOR} ".join(q["steps"]) for q in train], padding="max_length",
                       max_length=TOKEN_CAP, truncation=True, return_tensors="pt").input_ids
    assert (tokens == marker).sum(1).tolist() == [len(q["steps"]) - 1 for q in train]
    assert all(structure(row, tokenizer)[1] for row in tokens.tolist())
    target = tokens.masked_fill(tokens == vocab, -100).cuda()
    before = summarize(model, tokenizer, valid)
    lr = 0.00015
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    rng = Random(a.seed + 13507)
    by_depth = {n: [i for i, q in enumerate(train) if len(q["steps"]) == n] for n in HOPS}
    losses = []
    model.train()
    torch.cuda.synchronize()
    training = perf_counter()
    for step in range(a.steps):
        # Equal hop exposure with random samples from just the fit split.
        ids = [i for n in HOPS for i in rng.sample(by_depth[n], a.per_hop)]
        rng.shuffle(ids)
        index = torch.tensor(ids, dtype=torch.long, device="cuda")
        x = {k: v.index_select(0, index) for k, v in queries.items()}
        opt.param_groups[0]["lr"] = lr * min(1.0, (step + 1) / 40)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**x, labels=target[index]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (0, 39, 119, 239, 359, a.steps - 1):
            assert torch.isfinite(loss)
            point = {"step": step + 1, "loss": float(loss.detach())}
            losses.append(point)
            print(point, flush=True)
    torch.cuda.synchronize()
    training = perf_counter() - training
    after = summarize(model, tokenizer, valid)
    weights = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"weights": weights, "seed": a.seed, "data_sha256": digest_data,
                "fixture_sha256": panel["fixture_sha256"], "revision": MODEL_REVISION}, a.weights)
    result = {"data_sha256": digest_data, "fixture_sha256": panel["fixture_sha256"], "seed": a.seed,
              "revision": MODEL_REVISION, "seed_per_hop": a.per_hop, "steps": a.steps,
              "device": torch.cuda.get_device_name(), "training_seconds": training, "total_seconds": perf_counter() - start,
              "validation_base": before, "validation_fit": after, "losses": losses}
    a.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


def inference(a):
    cases = json.loads(a.fixture.read_text())["rows"]
    tokenizer, model = interfaces(a.assets)
    weights_hash = None
    if a.weights:
        weights_hash = digest(a.weights)
        trained = torch.load(a.weights, map_location="cpu", weights_only=True)
        assert trained["fixture_sha256"] == digest(a.fixture) and trained["revision"] == MODEL_REVISION
        model.load_state_dict(trained["weights"])
    cases = cases[a.shard::a.shards]
    prompts = [q["question"] for q in cases]
    results = predict(model, tokenizer, prompts)
    output = {"weights_sha256": weights_hash, "fixture_sha256": digest(a.fixture), "revision": MODEL_REVISION,
              "shard": a.shard, "shards": a.shards, "device": torch.cuda.get_device_name(),
              "rows": [{"id": q["id"], "steps": s, "valid": v}
                       for q, (s, v) in zip(cases, results, strict=True)]}
    a.output.write_bytes(gzip.compress(json.dumps(output, separators=(",", ":")).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "inference"))
    for name in ("assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("data", "manifest", "fixture", "weights"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--seed", type=int, default=106331)
    p.add_argument("--steps", type=int, default=480)
    p.add_argument("--per-hop", type=int, default=6)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision("highest")
    (fit if a.mode == "fit" else inference)(a)


if __name__ == "__main__":
    main()
