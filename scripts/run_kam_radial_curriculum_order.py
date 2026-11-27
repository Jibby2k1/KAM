#!/usr/bin/env python3
"""Compare early short training with interleaving at equal per-length exposure."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from benchmark_kam_radial_backward import objective
from benchmark_kam_radial_sdpa import predict
from run_kam_bitemporal_decoder import encode, fingerprint, make_decoder, pack, record_order
from run_kam_bitemporal_noise import generate
from run_kam_radial_dispatch import DISPATCH, assess, choose

CONFIG = ((16, 124, 4096), (48, 42, 4096), (128, 16, 2048), (512, 4, 1024))
EVALUATION = (20, 128, 512, 1024)
POLICIES = ("interleaved", "short_first")


def training_order(steps, control, seed):
    lengths = [d for d, _, _ in CONFIG]
    common = 32
    assert steps % 16 == 0 and steps // len(lengths) > common
    # Share exactly the warmup batches. Subsequently vary only the order
    # of the same per-length examples, at a common constant learning rate.
    first = lengths[:2] * common
    all_counts = {d: steps // len(lengths) for d in lengths}
    remaining = all_counts.copy()
    for depth in first:
        remaining[depth] -= 1
    rng = Random(seed + 18811)
    early = []
    if control == "short_first":
        # Each short length reaches half of its total exposure before
        # seeing a long training example.
        count = steps // 8 - common
        early = lengths[:2] * count
        for depth in early:
            remaining[depth] -= 1
    late = [d for d in lengths for _ in range(remaining[d])]
    rng.shuffle(late)
    result = first + early + late
    assert Counter(result) == all_counts
    return result


def setup(seed, steps):
    banks, held = {}, {}
    for depth, batch, count in CONFIG:
        scene = generate(seed + 424919 + depth, count, depth, "mixed", "cuda")
        order = record_order(count, depth, seed + 221293 + depth, "cuda")
        g = torch.Generator().manual_seed(seed + 622619 + depth)
        selected = torch.randint(count, (steps // len(CONFIG), batch), generator=g)
        stamp = fingerprint(scene, order), sha256(selected.numpy().tobytes()).hexdigest()
        banks[depth] = (scene, order, selected.cuda(), stamp)
    for depth in EVALUATION:
        count = 512 if depth <= 512 else 256
        panel = generate(seed + 981737 + depth, count, depth, "mixed", "cuda")
        order = record_order(count, depth, seed + 387019 + depth, "cuda")
        held[depth] = panel, order, fingerprint(panel, order)
    return banks, held


def evaluate(reference, model, held):
    reference.load_state_dict(model.state_dict())
    return assess(reference, "eager_fp32", held)


def fit(model, reference, banks, held, order, execution):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, fused=True)
    seen = Counter()
    tokens = 0
    checks = set((64, len(order)//8, len(order)//4, len(order)//2, 3*len(order)//4, len(order)))
    checkpoints = {}
    torch.cuda.synchronize()
    train_seconds = 0
    started = perf_counter()
    model.train()
    for j, depth in enumerate(order, 1):
        scene, perm, indices, _ = banks[depth]
        chosen = indices[seen[depth]]
        seen[depth] += 1
        x, entity, fact = encode(pack(scene, chosen), perm[chosen], hide_query_clock=True)
        path = choose(model, DISPATCH, depth) if execution == "selective" else "eager_fp32"
        optimizer.param_groups[0]["lr"] = 0.0004 * min(j / 64, 1)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(predict(model, path, x), entity, fact)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        tokens += x.shape[0] * x.shape[1]
        if j in checks:
            torch.cuda.synchronize()
            train_seconds += perf_counter() - started
            assert torch.isfinite(loss)
            checkpoints[str(j)] = {"tokens": tokens, "depth_updates": dict(seen), "loss": float(loss.detach()),
                                   "elapsed_update_seconds": train_seconds, "outcomes": evaluate(reference, model, held)}
            print(j, tokens, {d: sum(x and y for x, y in p["outcomes"])
                              for d, p in checkpoints[str(j)]["outcomes"].items()}, flush=True)
            torch.cuda.synchronize()
            started = perf_counter()
    assert seen == Counter(order)
    return checkpoints


def state_sha(model, *, common=False):
    return sha256(b"".join(v.cpu().contiguous().numpy().tobytes() for k, v in model.state_dict().items()
                           if not common or not k.endswith(("raw_metric", "raw_bandwidth")))).hexdigest()


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    assert a.steps >= 3200 and a.steps % 16 == 0
    banks, held = setup(a.seed, a.steps)
    assert a.execution == "eager" or a.geometry == "radial"
    arm = f"{a.geometry}_without_position"
    reference = make_decoder(arm, a.seed, "cuda")
    original = state_sha(reference)
    common = state_sha(reference, common=True)
    baseline = assess(reference, "eager_fp32", held)
    results = {}
    for control in (*POLICIES[a.order:], *POLICIES[:a.order]):
        model = make_decoder(arm, a.seed, "cuda")
        assert original == state_sha(model)
        order = training_order(a.steps, control, a.seed)
        digest = sha256(" ".join(map(str, order)).encode()).hexdigest()
        outcomes = fit(model, reference, banks, held, order, a.execution)
        results[control] = {"update_order_sha256": digest, "held_out": outcomes, "final_state_sha256": state_sha(model)}
        print(control, outcomes[str(a.steps)]["elapsed_update_seconds"], flush=True)
    a0, a1 = (r["held_out"] for r in results.values())
    assert a0["64"] == a1["64"] or all(a0["64"][k] == a1["64"][k] for k in ("tokens", "depth_updates", "loss", "outcomes"))
    assert a0[str(a.steps)]["tokens"] == a1[str(a.steps)]["tokens"]
    assert a0[str(a.steps)]["depth_updates"] == a1[str(a.steps)]["depth_updates"]
    return {"config": CONFIG, "device": torch.cuda.get_device_name(), "seed": a.seed, "steps": a.steps,
            "geometry": a.geometry, "execution": a.execution, "initial_shared_parameter_sha256": common,
            "initial_state_sha256": original, "banks": {str(d): b[-1] for d, b in banks.items()},
            "initial_validation": baseline, "policies": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=9600)
    p.add_argument("--geometry", choices=("radial", "dot"), default="radial")
    p.add_argument("--execution", choices=("selective", "eager"), default="selective")
    p.add_argument("--order", type=int, choices=range(2), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
