#!/usr/bin/env python3
"""Replicate learned top-four selection on long inputs and profile the full reader."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark_kam_sparse_live_read import dense, profile
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest, layers_for, load_registered_model, model_identity
from kam_exact_sparse_readout import exact_sparse_read
from kam_live_token_factorial import GPU, LiveTokenAdapter, LiveTokenStack
from kam_math_kl_protocol import write_json
from kam_pretrained_adapters import HEADS, WIDTH, mixed_precision
from prepare_kam_long_sparse import CONTEXT, PROTOCOL, TARGETS

RUNS = tuple((seed, arm) for seed in SEEDS for arm in ("soft-linear", "top4-linear")) + ((0, "pretrained"),)
STEP = 256
AUDIT_QUESTIONS = 24


class InferenceAdapter(LiveTokenAdapter):
    def forward(self, x):
        if self.backend == "explicit":
            return super().forward(x)
        if self.cache_enabled or self.cache is not None:
            raise ValueError("Full-reader benchmark expects prefill without KV cache")
        z = self.down(F.rms_norm(x, (x.shape[-1],)))
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        q = q + self.query_out(self.query_in(z))
        q, k, v = (r.unflatten(-1, (HEADS, WIDTH // HEADS)).transpose(1, 2) for r in (q, k, v))
        result = dense(q, k, v, True) if self.backend == "flash" else exact_sparse_read(q, k, v, causal=True)
        return x + self.up(self.out(result.transpose(1, 2).reshape_as(z))).to(x.dtype)


class InferenceStack(LiveTokenStack):
    def __init__(self, model, seed, arm):
        super().__init__(model, seed, "pretrained")
        if arm == "pretrained":
            return
        for i in layers_for(model):
            a = InferenceAdapter(model.config.hidden_size, arm, seed + i)
            self.modules.append(a)
            self.handles.append(model.model.layers[i].register_forward_hook(self.hook(a)))
        self.modules.cuda()
        self.modules.eval()

    def backend(self, name):
        for a in self.modules:
            a.backend = name


def prepare_run(root, source, work, seed, arm):
    manifest_path = work / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = work / "inputs.json"
    model_id, revision, _ = model_identity(root)
    assert (manifest["model"], manifest["revision"]) == (model_id, revision)
    assert manifest["protocol"] == PROTOCOL and manifest["inputs_sha256"] == digest(inputs)
    assert torch.cuda.get_device_name() == GPU
    model, tokenizer = load_registered_model(root)
    stack = InferenceStack(model, seed, arm)
    adapter_hash = None
    if arm != "pretrained":
        name = f"{seed}-{arm}"
        path = source / f"adapter-{name}-{STEP}.pt"
        receipt = json.loads((source / f"train-{name}.json").read_text())
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        adapter_hash = digest(path)
        assert receipt["history"][-1]["adapter_sha256"] == adapter_hash and checkpoint["step"] == STEP
        assert checkpoint["identity"]["seed"] == seed and checkpoint["identity"]["arm"] == arm
        stack.modules.load_state_dict(checkpoint["adapter"])
    labels = [tokenizer.encode(c, add_special_tokens=False) for c in ("A", "B")]
    assert all(len(ids) == 1 for ids in labels) and labels[0] != labels[1] and model.lm_head.bias is None
    proj = model.lm_head.weight.index_select(0, torch.tensor([x[0] for x in labels], device="cuda"))
    return model, stack, proj, json.loads(inputs.read_text()), adapter_hash


@torch.inference_mode()
def language_score(model, stack, data, limit=None):
    stack.context(None, False)
    books = []
    for book in data["pg19"][:limit]:
        nll = []
        for row in book["windows"]:
            ids = torch.tensor([row], device="cuda")
            assert ids.size(-1) == CONTEXT + 1
            with mixed_precision():
                hidden = model.model(input_ids=ids[:, :-1], use_cache=False).last_hidden_state
                output = model.lm_head(hidden[:, -TARGETS:]).float()
            cross_entropy = F.cross_entropy(output.squeeze(0), ids[0, -TARGETS:], reduction="mean")
            nll.append(float(cross_entropy))
        books.append({"id": book["id"], "nll": nll})
    return books


@torch.inference_mode()
def retrieval_score(model, stack, projection, data, limit=None):
    rows = data["retrieval"][:2*limit if limit else None]
    records = []
    for i in range(0, len(rows), 2):
        pair = rows[i:i+2]
        x = torch.tensor([r["ids"] for r in pair], device="cuda")
        assert x.size(-1) == CONTEXT
        logits, _ = next_logits(model, stack, x, torch.ones_like(x, dtype=torch.bool), projection=projection)
        for row, (a, b) in zip(pair, logits.tolist(), strict=True):
            d = b-a
            records.append({k: row[k] for k in ("id", "expected", "family")} |
                           {"margin": d, "correct": d != 0 and int(d > 0) == row["expected"]})
    return records


def audit_backend(model, stack, projection, data, backend):
    results = {}
    for name in ("explicit", backend):
        stack.backend(name)
        results[name] = {"pg19": language_score(model, stack, data, 4),
                         "retrieval": retrieval_score(model, stack, projection, data, AUDIT_QUESTIONS)}
    a, b = results["explicit"], results[backend]
    differences = [abs(x-y) for a_book, b_book in zip(a["pg19"], b["pg19"], strict=True)
                   for x, y in zip(a_book["nll"], b_book["nll"], strict=True)]
    def choice(r):
        return (r["margin"] > 0) - (r["margin"] < 0)
    total = sum(choice(x) != choice(y) for x, y in zip(a["retrieval"], b["retrieval"], strict=True))
    return {"nll_max_abs_delta": max(differences), "choice_differences": total, "nll_windows": len(differences),
            "presentations": len(a["retrieval"]), "passed": total <= 1 and max(differences) <= .003, "scores": results}


def evaluate(root, source, work, index):
    seed, arm = RUNS[index]
    model, stack, proj, data, checkpoint = prepare_run(root, source, work, seed, arm)
    backend = {"pretrained": "flash", "soft-linear": "flash", "top4-linear": "sparse"}[arm]
    audit = None if arm == "pretrained" else audit_backend(model, stack, proj, data, backend)
    scored_backend = backend if audit is None or audit["passed"] else "explicit"
    stack.backend(scored_backend)
    scores = {"pg19": language_score(model, stack, data), "retrieval": retrieval_score(model, stack, proj, data)}
    write_json(work / f"eval-{seed}-{arm}.json", {"manifest_sha256": digest(work / "manifest.json"), "arm": arm, "seed": seed,
               "gpu": GPU, "scored_backend": scored_backend, "checkpoint_sha256": checkpoint, "audit": audit, **scores})


def evaluate_reference(root, source, work, index):
    seed, arm = RUNS[index]
    prefix = f"{seed}-{arm}"
    prior_path = work / f"eval-{prefix}.json"
    prior = json.loads(prior_path.read_text())
    manifest_hash = digest(work / "manifest.json")
    assert (prior["seed"], prior["arm"], prior["manifest_sha256"]) == (seed, arm, manifest_hash)
    if prior["scored_backend"] == "explicit":
        return
    model, stack, proj, data, checkpoint = prepare_run(root, source, work, seed, arm)
    assert checkpoint == prior["checkpoint_sha256"]
    stack.backend("explicit")
    scores = {"pg19": language_score(model, stack, data), "retrieval": retrieval_score(model, stack, proj, data)}
    write_json(work / f"reference-{prefix}.json", {"manifest_sha256": manifest_hash, "arm": arm, "seed": seed,
               "gpu": GPU, "scored_backend": "explicit", "checkpoint_sha256": checkpoint, **scores})


@torch.inference_mode()
def full_profile(root, source, work):
    seed = SEEDS[0]
    model, stack, proj, data, checkpoint = prepare_run(root, source, work, seed, "top4-linear")
    sample = [token for row in data["pg19"][0]["windows"] for token in row[:-1]]
    trials = {}
    for width in (512, 2048, 4096, 8192):
        tokens = torch.tensor([sample[:width]], device="cuda")
        mask = torch.ones_like(tokens, dtype=torch.bool)
        def action(backend):
            stack.backend(backend)
            return next_logits(model, stack, tokens, mask, projection=proj)[0]
        trials[str(width)] = profile({name: lambda name=name: action(name) for name in ("flash", "sparse")}, repeats=15)
    write_json(work / "full-profile.json", {"gpu": GPU, "seed": seed, "checkpoint_sha256": checkpoint,
               "manifest_sha256": digest(work / "manifest.json"), "trials": trials,
               "model_parameters_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
               "adapter_parameters_bytes": sum(p.numel()*p.element_size() for p in stack.modules.parameters())})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("mode", choices=("evaluate", "profile", "evaluate-reference"))
    p.add_argument("index", type=int, nargs="?", choices=range(len(RUNS)))
    a = p.parse_args()
    if a.mode == "evaluate":
        evaluate(a.root, a.source, a.work, a.index)
    elif a.mode == "profile":
        full_profile(a.root, a.source, a.work)
    else:
        evaluate_reference(a.root, a.source, a.work, a.index)
