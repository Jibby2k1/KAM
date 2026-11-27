#!/usr/bin/env python3
"""Cross eager/fused FP32 attention with exact per-shape graph capture."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
from math import cos, pi
from pathlib import Path
from random import Random
from types import MethodType

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_guarded_fp32_sdpa import dot_sdpa
from benchmark_kam_radial_sdpa import augmented_forward
from benchmark_kam_radial_shape_graphs import train
from benchmark_kam_radial_training_graph import differences
from run_kam_bitemporal_decoder import make_decoder
from run_kam_clock_release_training import source_panels
from run_kam_radial_archive_training import assess
from run_kam_rank_cache_graph import merged
from run_kam_temporal_clock_encoding import LONG_SCHEDULE, STEPS, banks, evaluate, exposure_order, holdout_panels
from run_kam_typed_clock_guard import GuardDecoder


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    bank = banks(a.seed, "mixed", LONG_SCHEDULE)
    material, _, checks = merged(bank)
    order = exposure_order(LONG_SCHEDULE, a.seed, False)
    assert len(order) == STEPS
    curriculum = {"interleaved": order}
    if a.curriculum_reference:
        curriculum.update(shallow_half=exposure_order(LONG_SCHEDULE, a.seed, True),
                          shallow_all=sorted(order, key=lambda d: d > 48))
        assert all(sorted(plan) == sorted(order) for plan in curriculum.values())
        prior = json.loads(a.curriculum_reference.read_bytes())
        assert prior["seed"] == a.seed and prior["device"] == torch.cuda.get_device_name() == "NVIDIA L4"
        assert prior["training_sha256"] == {str(k): v[-1] for k, v in bank.items()}
    rates = torch.tensor([.002*min(i/100, 1)*(.2 + .8*(1+cos(pi*i/STEPS))/2) for i in range(1, STEPS+1)], device="cuda")
    held = holdout_panels(a.seed, (48, 128, 512, 1024))
    files, archives = source_panels(a)

    def reference_kernel(model, depth):
        return "eager_fp32"

    performance = {}
    for geometry in ("dot", "radial"):
        arch = geometry + "_without_position"
        original = make_decoder(arch, a.seed, "cpu")
        original_weights = {f"decoder.{k}": v for k, v in original.state_dict().items()}
        config = {"weights": original_weights}
        work = ([("fused_fp32", "shape_graph", label) for label in curriculum] if a.curriculum_reference else
                [(attention, mode, "interleaved") for attention in ("eager_attention", "fused_fp32")
                 for mode in ("eager", "shape_graph")])
        weights, moments, metrics = {}, {}, {}
        for attention, mode, policy in Random(a.seed+len(geometry)).sample(work, len(work)):
            label = policy if a.curriculum_reference else f"{attention}/{mode}"
            core = make_decoder(arch, a.seed, "cuda")
            model = GuardDecoder(core, "hard_clock_and_rank")
            fused = attention == "fused_fp32"
            plain = [block.context.forward for block in core.blocks]
            if fused:
                apply = dot_sdpa if geometry == "dot" else augmented_forward
                for block in core.blocks:
                    block.context.forward = MethodType(apply, block.context)
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if fused else nullcontext():
                weights[label], moments[label], measured = train(model, config, material, curriculum[policy], mode, rates=rates,
                                                                  kernel_selector=reference_kernel)
            for block, forward in zip(core.blocks, plain, strict=True):
                block.context.forward = forward
            metrics[label] = measured | {"heldout": evaluate(model, held),
                                        "archives": {name: assess(model, "eager_fp32", records) for name, records in archives.items()}}
            print(geometry, label, measured["wall_seconds"], flush=True)
        if a.curriculum_reference:
            expected = prior["results"][geometry]["arms"]["fused_fp32/shape_graph"]
            assert all(metrics["interleaved"][k] == expected[k] for k in ("heldout", "archives", "loss"))
            performance[geometry] = {"arms": metrics, "interleaved_archive_replay": True}
            continue
        parity = {}
        for attention in ("eager_attention", "fused_fp32"):
            base, captured = [f"{attention}/{m}" for m in ("eager", "shape_graph")]
            parity[attention] = {"weights": differences(weights[base], weights[captured]),
                                 "moments": differences(moments[base], moments[captured]),
                                 "same_final_loss": metrics[base]["loss"] == metrics[captured]["loss"],
                                 "same_heldout": metrics[base]["heldout"] == metrics[captured]["heldout"],
                                 "same_archive": metrics[base]["archives"] == metrics[captured]["archives"]}
        drift = differences(weights["eager_attention/eager"], weights["fused_fp32/eager"])
        performance[geometry] = {"arms": metrics, "capture_parity": parity, "weights_eager_vs_fused": drift}
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "steps": len(order), "schedule": LONG_SCHEDULE,
            "sequence_sha256": {k: sha256(json.dumps(v, separators=(",", ":")).encode()).hexdigest() for k, v in curriculum.items()},
            "training_sha256": {str(k): v[-1] for k, v in bank.items()}, "merged_batch_checks": checks,
            "archives_sha256": {k: sha256(v.read_bytes()).hexdigest() for k, v in files.items()}, "results": performance}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", required=True, type=int)
    for key in ("dallas", "unemployment", "industry", "later-unemployment", "later-industry", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--curriculum-reference", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
