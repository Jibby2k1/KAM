#!/usr/bin/env python3
"""Compare exact on-the-fly address rank with compact ranks for repeated training histories."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from kam_address_local_clock import local_rank
from run_kam_bitemporal_decoder import encode, make_decoder, pack
from run_kam_clock_release_training import ENCODING, LocalClockDecoder
from run_kam_radial_curriculum_order import state_sha
from run_kam_temporal_clock_encoding import ClockDecoder, SCHEDULE, banks, evaluate, exposure_order, fit, holdout_panels

SENTINEL = 255


@torch.no_grad()
def rank_cache(panels):
    result = {}
    for d, (scenes, permutation, _, _) in panels.items():
        assert d < SENTINEL
        grouped = []
        for scene in scenes:
            encoded = []
            for lower in range(0, len(permutation), 256):
                take = slice(lower, lower+256)
                tokens, _, _ = encode(pack(scene, take), permutation[take], hide_query_clock=True)
                ranks, observed = local_rank(tokens)
                assert bool(ranks[observed].lt(SENTINEL).all())
                encoded.append(ranks.masked_fill(~observed, SENTINEL).to(torch.uint8))
            grouped.append(torch.cat(encoded))
        result[d] = tuple(grouped)
    return result


def paste(tokens, selected):
    assert tokens.shape[:2] == selected.shape
    tokens[..., 39] = torch.where(selected.ne(SENTINEL), (selected.float()+1).reciprocal(), tokens[..., 39])
    return tokens


def inject(cache):
    def apply(tokens, _key, d, slots, groups):
        items = [values[indices] for values, indices in zip(cache[d], slots.chunk(groups), strict=True)]
        return paste(tokens, torch.cat(items))
    return apply


def verify(cache, panels):
    from kam_address_local_clock import local_features
    from run_kam_temporal_clock_encoding import recode

    checked = 0
    for d, (scenes, order, slots, _) in panels.items():
        for i in (0, len(slots)//2, len(slots)-1):
            sample = slots[i].chunk(len(scenes))
            data = torch.cat([encode(pack(scene, inds), order[inds], hide_query_clock=True)[0]
                              for scene, inds in zip(scenes, sample, strict=True)])
            with_updates = inject(cache)(data.clone(), None, d, slots[i], len(scenes))
            baseline = local_features(data, recode(data, ENCODING), alias=False, sham=False)
            assert torch.equal(baseline, recode(with_updates, ENCODING))
            checked += len(data)
    return checked


def run(seed):
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    train = banks(seed, "mixed", SCHEDULE)
    update = exposure_order(SCHEDULE, seed, False)
    holdout = holdout_panels(seed, (16, 48, 128, 512))
    torch.cuda.synchronize()
    start = perf_counter()
    cache = rank_cache(train)
    torch.cuda.synchronize()
    build = perf_counter() - start
    size = sum(x.nbytes for seq in cache.values() for x in seq)
    validated = verify(cache, train)
    choices = [(k, method) for k in ("dot", "radial") for method in ("on_the_fly", "cached_uint8")]
    validated_runs = {}
    for geometry, method in Random(seed + 6609).sample(choices, len(choices)):
        model = make_decoder(geometry + "_without_position", seed, "cuda")
        online = LocalClockDecoder(model, "fact_rank_feature")
        training = online if method == "on_the_fly" else ClockDecoder(model, ENCODING)
        transform = inject(cache) if method == "cached_uint8" else None
        tracked = fit(training, train, update, transform=transform, audit_optimizer=True)
        tested = evaluate(online, holdout)
        hash_ = state_sha(model)
        validated_runs[f"{geometry}/{method}"] = {"state_sha256": hash_, "training": tracked, "heldout": tested}
        print(geometry, method, tracked["seconds"], flush=True)
    for g in ("dot", "radial"):
        a, b = (validated_runs[f"{g}/{m}"] for m in ("on_the_fly", "cached_uint8"))
        assert a["training"]["losses"] == b["training"]["losses"]
        assert a["training"]["optimizer_sha256"] == b["training"]["optimizer_sha256"]
        assert a["state_sha256"] == b["state_sha256"]
        assert a["heldout"] == b["heldout"]
    return {"seed": seed, "device": torch.cuda.get_device_name(), "setup_seconds": build, "cache_bytes": size,
            "direct_tensor_checks": validated, "schedule": SCHEDULE,
            "updates": len(update), "clock": ENCODING,
            "training_banks": {str(d): row[-1] for d, row in train.items()},
            "order_sha256": sha256(json.dumps(update, separators=(",", ":")).encode()).hexdigest(),
            "results": validated_runs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.seed), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
