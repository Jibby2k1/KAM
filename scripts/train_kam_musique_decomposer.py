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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from audit_kam_natural_data import digest, words

REFERENCE = re.compile(r"#(\d+)\b")
SEPARATOR = "<extra_id_0>"
ATOMIC_REFERENCES = {i: f"<extra_id_{i}>" for i in (1, 2, 3)}
HOPS = (2, 3, 4)
TOKEN_CAP = 160
MODEL_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"


def instruction(q):
    return f"Break the question into short ordered subquestions. Refer to earlier answers with #1, #2, and #3. Question: {q}"


def normalized_tmpl(text):
    return " ".join(words(text))


def model_source(assets, size):
    if size == "base":
        from run_kam_musique_bindings import BASE_REVISION

        return assets / f"reader_base_{BASE_REVISION}", BASE_REVISION
    assert size == "small"
    model_config = json.loads((assets / "revisions.json").read_text())
    assert model_config["google/flan-t5-small"] == MODEL_REVISION
    return assets / "reader", MODEL_REVISION


def interfaces(assets, size="small"):
    reader, _ = model_source(assets, size)
    if size == "base":
        from run_kam_musique_bindings import BASE_WEIGHT

        assert digest(reader / "model.safetensors") == BASE_WEIGHT
    tokenizer = AutoTokenizer.from_pretrained(reader, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(reader, local_files_only=True, use_safetensors=True).cuda()
    return tokenizer, model


def structure(tokens, tokenizer, binding_tokens=False):
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
    symbolic = {tokenizer.convert_tokens_to_ids(value): f"#{i}" for i, value in ATOMIC_REFERENCES.items()} if binding_tokens else {}

    def decode(row):
        if not symbolic:
            return " ".join(tokenizer.decode(row, skip_special_tokens=True).split())
        fragments, start = [], 0
        for i, token in enumerate(row):
            if token in symbolic:
                fragments.extend((tokenizer.decode(row[start:i], skip_special_tokens=True), symbolic[token]))
                start = i + 1
        fragments.append(tokenizer.decode(row[start:], skip_special_tokens=True))
        return " ".join(" ".join(fragments).split())

    steps = [decode(row) for row in items]
    valid = finished and len(steps) in HOPS
    valid &= all(part and all(1 <= int(i) <= n for i in REFERENCE.findall(part))
                 for n, part in enumerate(steps))
    valid &= any(REFERENCE.search(p) for p in steps)
    return steps, bool(valid)


class RequiredBoundaries(LogitsProcessor):
    """Decode precisely the requested number of segments for the depth oracle."""

    def __init__(self, tokenizer, depths, device):
        self.separator = tokenizer.convert_tokens_to_ids(SEPARATOR)
        self.eos = tokenizer.eos_token_id
        self.boundaries = torch.tensor(depths, device=device) - 1

    def __call__(self, input_ids, scores):
        delimiter = input_ids == self.separator
        outstanding = self.boundaries - delimiter.sum(1)
        scores[:, self.separator].masked_fill_(outstanding <= 0, -torch.inf)
        scores[:, self.eos].masked_fill_(outstanding > 0, -torch.inf)
        positions = torch.arange(input_ids.shape[-1], device=input_ids.device)
        last_boundary = (delimiter * positions).amax(1)
        length = input_ids.shape[-1] - 1 - last_boundary
        force = (length >= 36) & (outstanding > 0)
        scores.masked_fill_(force[:, None], -torch.inf)
        scores[:, self.separator].masked_fill_(force, 0)
        return scores


class BackwardReferences(LogitsProcessor):
    """Admit a binding token only after the referenced step is complete."""

    def __init__(self, tokenizer):
        self.separator = tokenizer.convert_tokens_to_ids(SEPARATOR)
        self.tokens = {i: tokenizer.convert_tokens_to_ids(symbol) for i, symbol in ATOMIC_REFERENCES.items()}

    def __call__(self, input_ids, scores):
        steps = (input_ids == self.separator).sum(1)
        for i, token in self.tokens.items():
            scores[:, token].masked_fill_(steps < i, -torch.inf)
        return scores


@torch.inference_mode()
def predict(model, tokenizer, questions, batch=32, depths=None, *, binding_tokens=False, enforce_binding_order=False):
    model.eval()
    outputs = []
    for start in range(0, len(questions), batch):
        text = list(map(instruction, questions[start:start+batch]))
        value = tokenizer(text, padding=True, truncation=True, max_length=TOKEN_CAP, return_tensors="pt").to("cuda")
        constraints = LogitsProcessorList()
        if depths is not None:
            constraints.append(RequiredBoundaries(tokenizer, depths[start:start+batch], "cuda"))
        if enforce_binding_order:
            assert binding_tokens
            constraints.append(BackwardReferences(tokenizer))
        sample = model.generate(**value, max_new_tokens=TOKEN_CAP, do_sample=False, logits_processor=constraints)
        outputs.extend(structure(t, tokenizer, binding_tokens) for t in sample.tolist())
    return outputs


def summarize(model, tokenizer, rows, binding_tokens=False):
    predicted = predict(model, tokenizer, [q["question"] for q in rows], binding_tokens=binding_tokens)
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
    tokenizer, model = interfaces(a.assets, a.model_size)
    _, revision = model_source(a.assets, a.model_size)
    vocab = tokenizer.pad_token_id
    marker = tokenizer.convert_tokens_to_ids(SEPARATOR)
    queries = tokenizer([instruction(q["question"]) for q in train], padding="max_length",
                        max_length=TOKEN_CAP, truncation=True, return_tensors="pt").to("cuda")
    plans = [" " + f" {SEPARATOR} ".join(q["steps"]) for q in train]
    if a.binding_tokens:
        plans = [REFERENCE.sub(lambda m: f" {ATOMIC_REFERENCES[int(m[1])]} ", plan) for plan in plans]
    tokens = tokenizer(plans, padding="max_length",
                       max_length=TOKEN_CAP, truncation=True, return_tensors="pt").input_ids
    assert (tokens == marker).sum(1).tolist() == [len(q["steps"]) - 1 for q in train]
    assert all(structure(row, tokenizer, a.binding_tokens)[1] for row in tokens.tolist())
    target = tokens.masked_fill(tokens == vocab, -100).cuda()
    before = summarize(model, tokenizer, valid, a.binding_tokens)
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
    after = summarize(model, tokenizer, valid, a.binding_tokens)
    weights = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"weights": weights, "seed": a.seed, "data_sha256": digest_data,
                "fixture_sha256": panel["fixture_sha256"], "revision": revision, "binding_tokens": a.binding_tokens,
                "model_size": a.model_size}, a.weights)
    result = {"data_sha256": digest_data, "fixture_sha256": panel["fixture_sha256"], "seed": a.seed,
              "revision": revision, "seed_per_hop": a.per_hop, "steps": a.steps, "binding_tokens": a.binding_tokens,
              "parameters": sum(v.numel() for v in model.parameters()), "model_size": a.model_size,
              "device": torch.cuda.get_device_name(), "training_seconds": training, "total_seconds": perf_counter() - start,
              "validation_base": before, "validation_fit": after, "losses": losses}
    a.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


def inference(a):
    fixture = json.loads(a.fixture.read_text())
    cases = fixture["rows"]
    weights_hash = None
    binding_tokens = False
    trained = torch.load(a.weights, map_location="cpu", weights_only=True) if a.weights else None
    size = trained.get("model_size", "small") if trained else a.model_size
    tokenizer, model = interfaces(a.assets, size)
    _, revision = model_source(a.assets, size)
    if a.weights:
        weights_hash = digest(a.weights)
        source_fixture = a.trained_for or digest(a.fixture)
        assert trained["fixture_sha256"] == source_fixture and trained["revision"] == revision
        if a.trained_for:
            assert fixture["base_fixture_sha256"] == a.trained_for
            assert fixture["decomposer_data_sha256"] == trained["data_sha256"]
        model.load_state_dict(trained["weights"])
        binding_tokens = trained.get("binding_tokens", False)
    cases = cases[a.shard::a.shards]
    prompts = [q["question"] for q in cases]
    depths = [q["hops"] for q in cases] if a.oracle_depth else None
    results = predict(model, tokenizer, prompts, depths=depths, binding_tokens=binding_tokens,
                      enforce_binding_order=a.enforce_binding_order)
    output = {"weights_sha256": weights_hash, "fixture_sha256": digest(a.fixture), "revision": revision,
              "shard": a.shard, "shards": a.shards, "device": torch.cuda.get_device_name(),
              "rows": [{"id": q["id"], "steps": s, "valid": v}
                       for q, (s, v) in zip(cases, results, strict=True)]}
    if size != "small":
        output["model_size"] = size
    if a.oracle_depth:
        output["oracle_depth"] = True
    if binding_tokens:
        output.update(binding_tokens=True, enforce_binding_order=a.enforce_binding_order)
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
    p.add_argument("--oracle-depth", action="store_true")
    p.add_argument("--binding-tokens", action="store_true")
    p.add_argument("--enforce-binding-order", action="store_true")
    p.add_argument("--model-size", choices=("small", "base"), default="small")
    p.add_argument("--trained-for")
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
