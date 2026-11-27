#!/usr/bin/env python3
"""Benchmark physically shorter batches after exact causal clock admission."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_radial_precision import variants
from benchmark_kam_radial_sdpa import labels
from run_kam_bitemporal_decoder import encode, fingerprint, record_order
from run_kam_bitemporal_noise import exact, generate
from run_kam_temporal_clock_encoding import compact_tokens

EAGER, FLASH = "eager_fp32", "balanced_split_flash_bf16"
METHODS = ("full", "batch_max", "length_buckets")
SHAPES = ((128, 1), (128, 16), (512, 1), (512, 8), (1024, 1), (1024, 4))
BLOCK = 64


def modify_unrelated(scene, seed):
    q, aliases, facts, gold_a, gold_f, value = scene
    aliases, facts = aliases.clone(), facts.clone()
    depth = aliases.shape[1]
    target = aliases[..., 2].gather(1, gold_a[:, None])
    rng = torch.Generator(device=q.device).manual_seed(seed)
    probability = torch.tensor([0., .35, .65, .95], device=q.device)[torch.arange(len(q), device=q.device) % 4, None]
    foreign_alias = (aliases[..., 0] != q[:, :1]) | (aliases[..., 1] != q[:, 1:2])
    foreign_fact = (facts[..., 0] != q[:, :1]) | (facts[..., 1] != target)
    for records, foreign in ((aliases, foreign_alias), (facts, foreign_fact)):
        selected = foreign & (torch.rand(len(q), depth, device=q.device, generator=rng) < probability)
        records[..., 4] = torch.where(selected, q[:, 3:4] + 1, records[..., 4])
    a, f = exact(q, aliases, facts, "both")
    assert torch.equal(a, gold_a) and torch.equal(f, gold_f)
    return q, aliases, facts, gold_a, gold_f, value


def read(model, kernel, data, positions):
    flash = kernel == FLASH
    context = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if flash else nullcontext()
    with torch.autocast("cuda", torch.bfloat16, enabled=flash), context:
        hidden, _ = model.encode(data)
        row = torch.arange(len(data), device=data.device)
        return model.readout(hidden[row, positions])


def evaluate(model, kernel, tokens, method, *, preprocess=compact_tokens):
    data, query = preprocess(tokens)
    if method == "full":
        return read(model, kernel, data, query)
    if method == "batch_max":
        stop = int(query.max()) + 1
        return read(model, kernel, data[:, :stop], query)
    assert method == "length_buckets"
    groups = {}
    for i, idx in enumerate(query.tolist()):
        cap = min(data.shape[1], ((idx + BLOCK) // BLOCK) * BLOCK)
        groups.setdefault(cap, []).append(i)
    found, order = [], []
    for limit, group in groups.items():
        indices = torch.tensor(group, dtype=torch.long, device=data.device)
        found.append(read(model, kernel, data[:, :limit].index_select(0, indices), query[indices]))
        order.extend(group)
    reverse = torch.tensor(sorted(range(len(order)), key=order.__getitem__), device=data.device)
    return torch.cat(found).index_select(0, reverse)


def profile(call, repeats):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for _ in range(repeats):
        begin, finish = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        t0 = perf_counter()
        begin.record()
        call()
        finish.record()
        finish.synchronize()
        records.append((1000 * (perf_counter() - t0), begin.elapsed_time(finish)))
    return {"wall_ms": median(x for x, _ in records), "stream_ms": median(y for _, y in records),
            "extra_peak_bytes": torch.cuda.max_memory_allocated() - baseline, "repeats": records}


@torch.inference_mode()
def run(a):
    torch.set_float32_matmul_precision("highest")
    sha = sha256(a.checkpoint.read_bytes()).hexdigest()
    assert sha == a.sha256
    state = torch.load(a.checkpoint, weights_only=True, map_location="cpu")
    assert state["arm"] == "radial_without_position" and state["hide_query_clock"]
    networks = variants(state)
    sample = Random(a.seed + 54102)
    panels = {}
    for d in (128, 512, 1024):
        source = generate(a.seed + d, 64, d, "mixed", "cuda")
        order = record_order(64, d, a.seed + d + 1031, "cuda")
        for condition in ("original", "vary_rejection"):
            scene = source if condition == "original" else modify_unrelated(source, a.seed + d + 61581)
            tokens, entity, payload = encode(scene, order, hide_query_clock=True)
            answer = torch.stack((entity, payload), dim=-1)
            span = tokens[..., -2:].le(0).all(-1).sum(1)
            outputs = {}
            for kernel, batch in [(k, n) for k in (EAGER, FLASH) for depth, n in SHAPES if depth == d]:
                model = networks[kernel]
                compare = {}
                for method in METHODS:
                    prob = torch.cat([evaluate(model, kernel, x, method) for x in tokens.split(batch)]).float()
                    assert torch.isfinite(prob).all()
                    compare[method] = prob
                expected = compare["full"]
                check = {}
                timings = {}
                for version in sample.sample(list(METHODS), len(METHODS)):
                    observed = compare[version]
                    check[version] = {"match": int(labels(observed).eq(labels(expected)).all(-1).sum()),
                                      "joint": int(labels(observed).eq(answer).all(-1).sum()),
                                      "max_delta": float((observed - expected).abs().max())}
                    # Rotate representative batches across seeds and depths.
                    offset = (a.seed % (64 // batch)) * batch
                    data = tokens[offset:offset+batch]
                    timings[version] = profile(lambda: evaluate(model, kernel, data, version), a.repeats)
                outputs[f"{kernel}/{batch}"] = {"synchronized_timings": timings, "parity": check,
                                                "timed_eligible": span[offset:offset+batch].tolist()}
            panels[f"{d}/{condition}"] = {"shape": list(tokens.shape), "eligible_counts": span.tolist(),
                                         "fixture_sha256": fingerprint(scene, order), "profiles": outputs}
            print(d, condition, flush=True)
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "checkpoint_sha256": sha,
            "torch": str(torch.__version__), "panels": panels}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--sha256", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
