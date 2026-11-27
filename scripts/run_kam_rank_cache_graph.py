#!/usr/bin/env python3
"""Cross rank-cache reuse with CUDA Graph capture while checking exact training trajectories."""

import argparse
from hashlib import sha256
import json
from math import cos, pi
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from benchmark_kam_radial_shape_graphs import train
from benchmark_kam_radial_training_graph import differences
from run_kam_bitemporal_decoder import encode, make_decoder, pack
from run_kam_clock_release_training import ENCODING, LocalClockDecoder
from run_kam_rank_feature_cache import paste, rank_cache, verify
from run_kam_temporal_clock_encoding import ClockDecoder, SCHEDULE, STEPS, banks, evaluate, exposure_order, holdout_panels

POLICIES = (("eager", "on_the_fly"), ("eager", "cached_uint8"),
            ("shape_graph", "on_the_fly"), ("shape_graph", "cached_uint8"))


def merged(banks_by_depth, feature_cache=None):
    banks, ranks = {}, {}
    checked = 0
    for depth, (scenes, order, batches, seal) in banks_by_depth.items():
        count = len(order)
        n = len(scenes)
        assert all(len(scene[0]) == count for scene in scenes)
        assert batches.shape[1] % n == 0
        places = torch.cat([b + i * count for i, b in enumerate(batches.chunk(n, dim=1))], dim=1)
        full = tuple(torch.cat(v) for v in zip(*scenes, strict=True))
        repeated = order.repeat(n, 1)
        for slot in (0, len(batches)//2, len(batches)-1):
            selected = batches[slot]
            reference = [encode(pack(scene, index), order[index], hide_query_clock=True)
                         for scene, index in zip(scenes, selected.chunk(n), strict=True)]
            proposed = encode(pack(full, places[slot]), repeated[places[slot]], hide_query_clock=True)
            assert all(torch.equal(t, torch.cat(v)) for t, v in zip(proposed, zip(*reference, strict=True), strict=True))
            checked += len(selected)
        banks[depth] = full, repeated, places, seal
        if feature_cache is not None:
            ranks[depth] = torch.cat(feature_cache[depth])
    return banks, ranks, checked


def run(seed):
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    panels = banks(seed, "mixed", SCHEDULE)
    order = exposure_order(SCHEDULE, seed, False)
    assert len(order) == STEPS
    rates = torch.tensor([.002 * min(i/100, 1) * (.2 + .8 * (1 + cos(pi*i/STEPS))/2) for i in range(1, STEPS+1)], device="cuda")
    heldout = holdout_panels(seed, (16, 48, 128, 512))
    torch.cuda.synchronize()
    start = perf_counter()
    original_cache = rank_cache(panels)
    torch.cuda.synchronize()
    build_seconds = perf_counter() - start
    exact_cache_checks = verify(original_cache, panels)
    material, cache, check = merged(panels, original_cache)
    size = sum(v.nbytes for v in cache.values())

    def cache_prepare(tokens, d, indices):
        return paste(tokens, cache[d][indices])

    def eager_kernel(model, depth):
        return "eager_fp32"

    output = {}
    for g in ("dot", "radial"):
        runs, states, opts = {}, {}, {}
        for execution, feature in Random(seed + len(g)).sample(POLICIES, len(POLICIES)):
            key = f"{execution}/{feature}"
            arch = g + "_without_position"
            base, cpu = (make_decoder(arch, seed, d) for d in ("cuda", "cpu"))
            online = LocalClockDecoder(base, "fact_rank_feature")
            from_cache = feature == "cached_uint8"
            model = ClockDecoder(base, ENCODING) if from_cache else online
            seed_state = {f"decoder.{k}": value for k, value in cpu.state_dict().items()}
            weights, moments, stats = train(model, {"weights": seed_state}, material, order, execution,
                                            rates=rates, kernel_selector=eager_kernel,
                                            prepare_batch=cache_prepare if from_cache else None)
            states[key], opts[key] = weights, moments
            runs[key] = stats | {"heldout": evaluate(online, heldout)}
            print(g, key, stats["wall_seconds"], flush=True)
        reference = "eager/on_the_fly"
        parity = {}
        for key in runs:
            parity[key] = {"weights": differences(states[reference], states[key]),
                           "adamw_moments": differences(opts[reference], opts[key]),
                           "same_loss": runs[reference]["loss"] == runs[key]["loss"],
                           "same_decisions": runs[reference]["heldout"] == runs[key]["heldout"]}
            assert parity[key]["weights"]["equal"] and parity[key]["adamw_moments"]["equal"]
            assert parity[key]["same_loss"] and parity[key]["same_decisions"]
        output[g] = {"runs": runs, "parity": parity}
    return {"seed": seed, "device": torch.cuda.get_device_name(), "cache_bytes": size, "cache_build_seconds": build_seconds,
            "cache_checks": exact_cache_checks, "merged_batch_checks": check, "steps": STEPS,
            "sequence_sha256": sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest(), "arms": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.seed), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
