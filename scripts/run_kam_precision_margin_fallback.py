#!/usr/bin/env python3
"""Audit low-precision margin fallback with calibration held apart from measured histories."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_radial_precision import variants
from benchmark_kam_radial_sdpa import labels
from run_kam_bitemporal_decoder import encode, fingerprint, record_order
from run_kam_bitemporal_noise import generate
from run_kam_final_row_attention import infer, prefix, state, terminal_output
from run_kam_packed_clock_inference import profile
from run_kam_radial_archive_training import archive_panels
from run_kam_temporal_clock_encoding import recode
from run_kam_temporal_correction_curriculum import intervention

POLICIES = ("eager", "flash", "precise_readout", "calibrated_fallback")
DEPTHS = ((16, 32), (48, 32), (128, 16), (512, 8), (1024, 4))


def slices(logits):
    return logits[:, :64], logits[:, 64:]


def margins(logits):
    parts = [segment.topk(2, dim=-1).values for segment in slices(logits.float())]
    return torch.stack([v[:, 0] - v[:, 1] for v in parts], dim=1)


def precision(
    networks, tokens, mode, max_error,
):
    if mode == "eager":
        return infer(networks, tokens, "readout_fp32")[0], None
    if mode == "precise_readout":
        flash = networks["balanced_split_flash_bf16"]
        with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            early = prefix(flash, tokens)
        return terminal_output(networks["eager_fp32"], early.float(), "readout_fp32")[0], None
    result = infer(networks, tokens, "full_flash")[0].float()
    if mode == "flash":
        return result, None
    assert mode == "calibrated_fallback"
    pending = (margins(result) <= 2 * max_error).any(dim=-1)
    indexes = torch.where(pending)[0]
    if len(indexes):
        chosen = tokens.index_select(0, indexes)
        precise, _ = infer(networks, chosen, "readout_fp32")
        result.index_copy_(0, indexes, precise.float())
    return result, pending


def scenes(noise_seed, d, count, clock, split):
    scene = generate(noise_seed + split + d, count, d, "mixed", "cuda")
    changed, _ = intervention(scene)
    order = record_order(len(scene[0]), d, noise_seed + 42 * split + d, "cuda")
    found = {}
    for condition, active in ((False, scene), (True, changed)):
        tokens, entity, value = encode(active, order, hide_query_clock=True)
        found[condition] = recode(tokens, clock), torch.stack((entity, value), -1), fingerprint(active, order)
    return found


def calibrate(networks, sets, batch):
    evidence, maxima, coverage = [], [], 0
    for tokens, _, stamp in sets.values():
        evidence.append(stamp)
        for group in tokens.split(batch):
            eager, flash = [precision(networks, group, m, None)[0].float() for m in ("eager", "flash")]
            maxima.append(torch.stack([p.abs().amax() for p in slices(eager-flash)]))
            coverage += len(group)
    return torch.stack(maxima).amax(0), evidence, coverage


def measure(networks, tokens, target, width, errors, *, timing_seed=None):
    timing = {}
    if timing_seed is not None:
        for mode in Random(timing_seed).sample(list(POLICIES), len(POLICIES)):
            timing[mode] = profile(lambda: precision(networks, tokens[:width], mode, errors), 24)
    output = {}
    reference = None
    for mode in POLICIES:
        values, asked = [], []
        for block in tokens.split(width):
            logits, mask = precision(networks, block, mode, errors)
            values.append(logits.float())
            if mask is not None:
                asked.append(mask)
        final = torch.cat(values)
        guess = labels(final)
        if reference is None:
            reference = final
        flags = guess.eq(target)
        same = guess.eq(labels(reference)).all(-1)
        difference = [float(v.abs().max()) for v in slices(final-reference)]
        report = {"full_match": same.tolist(), "correct_keys": flags[:, 0].tolist(),
                  "correct_values": flags[:, 1].tolist(), "max_logit_differences": difference}
        if asked:
            report["fallback"] = torch.cat(asked).tolist()
        if mode in timing:
            report["timing"] = timing[mode]
        output[mode] = report
    return output


@torch.inference_mode()
def test_run(a, networks, clock):
    calibration, results, archives, bounds = {}, {}, {}, {}
    for depth, width in DEPTHS:
        material = scenes(a.seed, depth, 96, clock, 839591)
        tested = scenes(a.seed, depth, 192, clock, 840793)
        max_error, ids, n = calibrate(networks, material, width)
        bounds[depth] = max_error
        calibration[str(depth)] = {"max_logit_error": max_error.tolist(), "tests": n, "fixture_hashes": ids}
        for perturbed, (tokens, target, data_hash) in tested.items():
            assert data_hash not in ids
            label = ("corrected" if perturbed else "original") + "/" + str(depth)
            results[label] = {"fixture_sha256": data_hash, "test": measure(networks, tokens, target, width, max_error,
                                                                        timing_seed=None if perturbed else a.seed+depth)}
        print(depth, calibration[str(depth)]["max_logit_error"], flush=True)
    for name, file in (("dallas", a.dallas), ("eurostat", a.eurostat)):
        outputs = []
        for panel in archive_panels(json.loads(file.read_bytes()), a.seed):
            x, key, value = panel["data"]
            rows = [{k: row[k] for k in ("id", "period")} for row in panel["rows"]]
            result = measure(networks, recode(x, clock), torch.stack((key, value), -1), 32, bounds[panel["depth"]])
            outputs.append({"depth": panel["depth"], "recode": panel["recode"], "fixture_sha256": panel["fixture_sha256"],
                            "rows": rows, "test": result})
        archives[name] = outputs
    return calibration, results, archives


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    checkpoint, (clock, fit, hash_) = state(a)
    nets = variants(checkpoint)
    calibration, tests, archives = test_run(a, nets, clock)
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "clock": clock, "fit": fit,
            "reference_sha256": hash_, "archives_sha256": {k: sha256(getattr(a, k).read_bytes()).hexdigest()
                                                          for k in ("dallas", "eurostat")},
            "calibration": calibration, "procedural": tests, "archive": archives}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("dallas", "eurostat", "output", "fit-reference"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
