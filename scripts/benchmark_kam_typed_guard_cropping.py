#!/usr/bin/env python3
"""Time physical suffix removal on replayed hard-clock and address-rank models."""

import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import torch

from benchmark_kam_radial_sdpa import labels
from run_kam_bitemporal_decoder import encode, fingerprint, make_decoder, record_order
from run_kam_bitemporal_noise import generate
from run_kam_packed_clock_inference import EAGER, METHODS, evaluate as cropped, modify_unrelated, profile
from run_kam_temporal_clock_encoding import SCHEDULE, banks, evaluate, exposure_order, fit, holdout_panels
from run_kam_typed_clock_guard import GuardDecoder, guarded_tokens

PANELS = ((128, 16), (512, 8), (1024, 4))


@torch.inference_mode()
def measure(seed, model, trials):
    results = {}
    preprocess = partial(guarded_tokens, ranked=True)
    rng = Random(seed + 110750)
    for depth, width in PANELS:
        base = generate(seed + depth + 127830, 64, depth, "mixed", "cuda")
        perm = record_order(64, depth, seed + depth + 121575, "cuda")
        for panel, scenes in (("standard", base), ("future_distractors", modify_unrelated(base, seed + 110639 + depth))):
            raw, entity, value = encode(scenes, perm, hide_query_clock=True)
            truth = torch.stack((entity, value), -1)
            reference = torch.cat([cropped(model, EAGER, x, "full", preprocess=preprocess) for x in raw.split(width)])
            compare = {}
            for variant in rng.sample(list(METHODS), len(METHODS)):
                logits = torch.cat([cropped(model, EAGER, x, variant, preprocess=preprocess) for x in raw.split(width)])
                found = labels(logits)
                check = {"same_decisions": found.eq(labels(reference)).all(-1).tolist(),
                         "strict": found.eq(truth).all(-1).tolist(), "max_logit_error": float((logits-reference).abs().max())}
                pos = (seed % (len(raw)//width)) * width
                sample = raw[pos:pos+width]
                compare[variant] = {"provenance": check, "timing": profile(
                    lambda: cropped(model, EAGER, sample, variant, preprocess=preprocess), trials)}
            span = raw[..., -2:].le(0).all(-1).sum(1)
            results[f"{depth}/{panel}"] = {"trials": trials, "fixture_sha256": fingerprint(scenes, perm),
                                          "batch": width, "timed_lengths": span[pos:pos+width].tolist(),
                                          "baseline_preprocess_timing": profile(lambda: preprocess(sample), trials),
                                          "arms": compare}
    return results


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    former = json.loads(a.reference.read_bytes())
    assert former["seed"] == a.seed and former["device"] == torch.cuda.get_device_name() == "NVIDIA L4"
    training = banks(a.seed, "mixed", SCHEDULE)
    perm = exposure_order(SCHEDULE, a.seed, False)
    hold = holdout_panels(a.seed, (16, 48, 128, 512, 1024))
    results = {}
    for geometry in ("dot", "radial"):
        model = make_decoder(f"{geometry}_without_position", a.seed, "cuda")
        controller = GuardDecoder(model, "hard_clock_and_rank")
        trained = fit(controller, training, perm)
        reference = former["arms"][f"{geometry}/hard_clock_and_rank"]
        assert trained["losses"] == reference["fit"]["losses"]
        assessed = evaluate(controller, hold)
        assert assessed == reference["generated"]
        results[geometry] = {"archived_loss_and_heldout_parity": True, "panels": measure(a.seed, model, a.repeats)}
        print(geometry, len(results[geometry]["panels"]), flush=True)
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "reference": sha256(a.reference.read_bytes()).hexdigest(),
            "arms": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
