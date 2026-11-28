#!/usr/bin/env python3
"""Measure omitted-block summaries on fixed dense SmolLM2 attention states."""
import argparse
import base64
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.llama.modeling_llama import repeat_kv

PROTOCOL = "kam-omitted-summary-v1"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
WEIGHTS = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
TOKENIZER = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
CONFIG = "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843"
BOOKS = 16
CONTEXT = 4096
BLOCK = 64
GROUP = 16
TILE = 32
SLOTS = 8


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def means(states, width):
    block = states.unflatten(1, (-1, width))
    return block.mean(dim=2)


def protected_blocks(representative, key_means, current):
    group_scores = representative @ key_means.transpose(1, 2)
    block_scores = group_scores.unflatten(-1, (-1, BLOCK // GROUP)).amax(dim=-1)
    block_ids = torch.arange(block_scores.size(-1), device=block_scores.device)
    flexible = (block_ids < current[:, None] - 1) & (block_ids != 0)
    block_scores = block_scores.masked_fill(~flexible, -torch.inf)
    chosen = torch.argsort(block_scores, descending=True, stable=True)[..., :SLOTS - 3]
    locked = (block_ids == current[:, None]) | (block_ids == current[:, None] - 1) | (block_ids == 0)
    return locked.expand_as(block_scores).clone().scatter_(-1, chosen, True)


def mixing_weight(retained_partition, log_group_mass):
    return torch.cat((retained_partition, log_group_mass), dim=-1).softmax(dim=-1)


def estimate(exact, retained_partition, log_group_mass, summary_values):
    weights = mixing_weight(retained_partition, log_group_mass)
    return (weights[..., :1] * exact + weights[..., 1:] @ summary_values,
            weights[..., 0])


def local_outputs(queries, keys, values, positions, scaling):
    length = keys.size(1)
    query = queries[:, positions]
    representatives = queries[:, positions // TILE * TILE]
    current = positions // BLOCK
    key_means = means(keys, GROUP)
    chosen = protected_blocks(representatives, key_means, current)
    assert (chosen.sum(dim=-1) == SLOTS).all()
    is_past = torch.arange(length) <= positions[:, None]
    keep_tokens = chosen.repeat_interleave(BLOCK, dim=-1) & is_past
    logits = (query * scaling) @ keys.transpose(1, 2)
    before_query = logits.masked_fill(~is_past, -torch.inf)
    kept_logits = before_query.masked_fill(~keep_tokens, -torch.inf)
    dense = before_query.softmax(dim=-1) @ values
    exact = kept_logits.softmax(dim=-1) @ values
    retained_partition = kept_logits.logsumexp(dim=-1, keepdim=True)
    dense_partition = before_query.logsumexp(dim=-1)
    actual_mass = (retained_partition.squeeze(-1) - dense_partition).exp()
    earlier_than_latest = torch.arange(length // BLOCK) < current[:, None] - 1
    omitted_block = ~chosen & earlier_than_latest
    assert omitted_block.any(dim=-1).all()
    omitted_16 = omitted_block.repeat_interleave(BLOCK // GROUP, dim=-1)
    summary_value = means(values, GROUP)
    key_residual = keys.unflatten(1, (-1, GROUP)) - key_means.unsqueeze(2)
    variance_diagonal = key_residual.square().mean(dim=2)
    diagonal_variance = (query * scaling).square() @ variance_diagonal.transpose(1, 2)
    log_mean_16 = math.log(GROUP) + (query * scaling) @ key_means.transpose(1, 2)
    log_mean_8 = math.log(GROUP // 2) + (query * scaling) @ means(keys, GROUP // 2).transpose(1, 2)
    log_true_16 = logits.unflatten(-1, (-1, GROUP)).logsumexp(dim=-1)
    variants = {
        "mean16": (log_mean_16, omitted_16, summary_value),
        "diagonal16": (log_mean_16 + torch.log1p(diagonal_variance / 2), omitted_16, summary_value),
        "mean8": (log_mean_8, omitted_16.repeat_interleave(2, dim=-1), means(values, GROUP // 2)),
        "exact_group_mass16": (log_true_16, omitted_16, summary_value),
    }
    outputs = {"protected": exact}
    mass_error = {}
    for name, (partition, omitted, value) in variants.items():
        output, estimated_mass = estimate(exact, retained_partition, partition.masked_fill(~omitted, -torch.inf), value)
        outputs[name] = output
        mass_error[name] = float((estimated_mass - actual_mass).abs().mean())
    assert mass_error["exact_group_mass16"] < 2e-5
    return dense, outputs, actual_mass, mass_error


def fixture_gate():
    generator = torch.Generator().manual_seed(824044)
    groups = torch.randn((2, 40, 16), dtype=torch.float64, generator=generator)
    keys = groups.repeat_interleave(GROUP, dim=1)
    values = torch.randn(keys.shape, dtype=keys.dtype, generator=generator)
    queries = torch.randn(keys.shape, dtype=keys.dtype, generator=generator)
    dense, variants, _, errors = local_outputs(queries, keys, values, torch.tensor([544, 608]), 0.25)
    assert not torch.allclose(dense, variants["protected"])
    for name in ("mean16", "diagonal16", "mean8", "exact_group_mass16"):
        assert float((dense - variants[name]).abs().max()) < 1e-11 and errors[name] < 1e-11


def dataset(tokenizer, source_path, book_dir, *, limit):
    source = json.loads(source_path.read_text())
    assert source["protocol"] == PROTOCOL
    excluded = set(source["excluded_ids"])
    allowed = sorted(source["eligible"], key=lambda row: sha(f"{PROTOCOL}/{row['name']}".encode()))
    assert allowed == source["eligible"] and len({row["name"] for row in allowed}) == len(allowed)
    selected = []
    for book in allowed:
        name = Path(book["name"])
        assert name.parent == Path("test") and name.stem not in excluded
        raw = (book_dir / name.name).read_bytes()
        assert int(book["size"]) == len(raw)
        assert base64.b64encode(hashlib.md5(raw).digest()).decode() == book["md5Hash"]
        tokens = tokenizer.encode(raw.decode(errors="replace"), add_special_tokens=False, verbose=False)
        if len(tokens) < CONTEXT:
            continue
        starts = range(CONTEXT, len(tokens) - CONTEXT + 1, CONTEXT)
        start = min(starts, key=lambda i: sha(f"{PROTOCOL}/{name.stem}/{i}".encode()), default=0)
        tokens = tokens[start:start + CONTEXT]
        selected.append(({"id": name.stem, "offset": start, "source_sha256": sha(raw),
                          "tokens_sha256": sha(np.asarray(tokens, dtype="<i4").tobytes())}, tokens))
        if len(selected) == limit:
            break
    assert len(selected) == limit
    return selected


def observe_dense(model, tokens, callback, *, tag):
    def observed(module, query, key, value, mask, scaling, **kwargs):
        dense, _ = sdpa_attention_forward(module, query, key, value, mask, scaling=scaling, **kwargs)
        keys = repeat_kv(key, module.num_key_value_groups)[0]
        values = repeat_kv(value, module.num_key_value_groups)[0]
        callback(module, query[0], keys, values, dense[0], scaling)
        return dense, None

    AttentionInterface.register(tag, observed)
    original = model.config._attn_implementation
    model.config._attn_implementation = tag
    try:
        model.model(torch.tensor([tokens]), use_cache=False)
    finally:
        model.config._attn_implementation = original


def audit(model, tokens, positions):
    results = {}

    def compare(module, query, keys, values, dense, scaling):
        reference, outputs, mass, mass_errors = local_outputs(query, keys, values, positions, scaling)
        difference = float((dense[positions].transpose(0, 1) - reference).abs().max())
        assert torch.allclose(dense[positions], reference.transpose(0, 1), atol=3e-4, rtol=3e-4), difference
        def project(heads):
            return module.o_proj(heads.transpose(0, 1).flatten(1))

        ref_projected = project(reference)
        energy = float(ref_projected.square().sum(dtype=torch.float64))
        variant_error = {name: float((project(output) - ref_projected).square().sum(dtype=torch.float64))
                         for name, output in outputs.items()}
        results[module.layer_idx] = {"dense_energy": energy, "projected_error": variant_error,
                                     "retained_mass_mean": float(mass.mean()), "mass_errors": mass_errors,
                                     "manual_sdpa_max_abs": difference}

    observe_dense(model, tokens, compare, tag=PROTOCOL)
    assert sorted(results) == list(range(model.config.num_hidden_layers))
    return [results[layer] for layer in sorted(results)]


def aggregate(books):
    policy_error = defaultdict(list)
    for book in books:
        rows = book["layers"]
        energy = sum(row["dense_energy"] for row in rows)
        for name in rows[0]["projected_error"]:
            policy_error[name].append(sum(row["projected_error"][name] for row in rows) / energy)
    base = np.array(policy_error["protected"])
    draw = np.random.default_rng(612570).integers(len(base), size=(20_000, len(base)))
    metrics = {}
    for name, values in policy_error.items():
        errors = np.array(values)
        delta = base - errors
        interval = np.quantile((delta[draw].mean(axis=1) / base[draw].mean(axis=1)), (.025, .975))
        metrics[name] = {"mean_relative_projected_error": float(errors.mean()), "relative_error_by_book": values,
                         "improvement_relative_to_protected": float(delta.mean() / base.mean()),
                         "improvement_books": int((delta > 0).sum()), "paired_book_bootstrap95": interval.tolist()}
    return metrics


def load_model(path):
    for name, expected in (("model.safetensors", WEIGHTS), ("tokenizer.json", TOKENIZER), ("config.json", CONFIG)):
        assert sha((path / name).read_bytes()) == expected
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32,
                                                attn_implementation="sdpa").eval()
    return model, tokenizer


@torch.inference_mode()
def main(args):
    torch.set_num_threads(args.threads)
    fixture_gate()
    model, tokenizer = load_model(args.model)
    positions = torch.arange(544, CONTEXT, BLOCK)
    books = []
    for record, tokens in dataset(tokenizer, args.sources, args.books, limit=args.limit):
        books.append({**record, "layers": audit(model, tokens, positions)})
        print("verified book", record["id"], flush=True)
    write_json(args.output, {"protocol": PROTOCOL, "revision": REVISION, "model_sha256": WEIGHTS,
                            "tokenizer_sha256": TOKENIZER, "config_sha256": CONFIG,
                            "source_list_sha256": sha(args.sources.read_bytes()), "positions": positions.tolist(),
                            "torch": torch.__version__, "books": books, "policies": aggregate(books)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "sources", "books", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--limit", type=int, default=BOOKS, choices=(1, BOOKS))
    parser.add_argument("--threads", type=int, default=4)
    main(parser.parse_args())
