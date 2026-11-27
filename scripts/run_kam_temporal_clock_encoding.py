#!/usr/bin/env python3
"""Test clock feature scaling and an explicit eligibility guard on correction histories."""

import argparse
from hashlib import sha256
import json
from math import cos, log1p, pi
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from benchmark_kam_radial_backward import objective
from benchmark_kam_radial_sdpa import labels
from run_kam_bitemporal_decoder import encode, fingerprint, make_decoder, pack, record_order
from run_kam_bitemporal_noise import generate
from run_kam_radial_archive_training import archive_panels, assess as read_archive
from run_kam_radial_curriculum_order import state_sha
from run_kam_temporal_correction_curriculum import intervention

VARIANTS = ("linear", "signed_log", "signed_log_guard")
COMPACTION = ("signed_log", "signed_log_guard", "signed_log_compact")
SCHEDULE = ((16, 100), (48, 34))
STEPS = 3200


def recode(tokens, mode):
    if mode == "linear":
        return tokens
    clock = tokens[..., -2:]
    values = clock.sign() * (clock.abs() * 24).log1p() / log1p(24)
    output = torch.cat((tokens[..., :-2], values), dim=-1)
    if mode == "signed_log_guard":
        available = clock.le(0).all(-1, keepdim=True)
        output = output.masked_fill(~available, 0)
    return output


class ClockDecoder(torch.nn.Module):
    def __init__(self, decoder, mode):
        super().__init__()
        self.decoder = decoder
        self.mode = mode

    def forward(self, tokens):
        if self.mode != "signed_log_compact":
            return self.decoder(recode(tokens, self.mode))
        # The final query is the last eligible token. Move rejected events
        # after it so causal attention cannot see them at any layer.
        eligible = tokens[..., -2:].le(0).all(-1)
        order = (~eligible).to(torch.int32).argsort(dim=1, stable=True)
        data = recode(tokens, "signed_log").gather(1, order.unsqueeze(-1).expand_as(tokens))
        hidden, _ = self.decoder.encode(data)
        query = eligible.sum(dim=1) - 1
        return self.decoder.readout(hidden[torch.arange(len(hidden), device=hidden.device), query])


def banks(seed):
    by_depth = {}
    for d, count in SCHEDULE:
        original = generate(seed + 681133 + d, 4096, d, "mixed", "cuda")
        scene, changed = intervention(original)
        order = record_order(len(scene[0]), d, seed + 772709 + d, "cuda")
        g = torch.Generator().manual_seed(seed + 575019 + d)
        sample = torch.randint(len(scene[0]), (STEPS // 2, count), generator=g)
        by_depth[d] = scene, order, sample.cuda(), {"scene_sha256": fingerprint(scene, order),
                                                   "sample_sha256": sha256(sample.numpy().tobytes()).hexdigest(),
                                                   "records_changed": changed}
    return by_depth


def fit(model, panels):
    opt = torch.optim.AdamW(model.parameters(), lr=.002, fused=True)
    model.train()
    torch.cuda.synchronize()
    started = perf_counter()
    losses = []
    for k in range(STEPS):
        d, _ = SCHEDULE[k % len(SCHEDULE)]
        scene, order, batches, _ = panels[d]
        slots = batches[k // len(SCHEDULE)]
        tokens, key, value = encode(pack(scene, slots), order[slots], hide_query_clock=True)
        step = k + 1
        opt.param_groups[0]["lr"] = .002 * min(step / 100, 1) * (.2 + .8 * (1 + cos(pi * step / STEPS)) / 2)
        opt.zero_grad(set_to_none=True)
        loss = objective(model(tokens), key, value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step in (100, 400, 800, 1600, STEPS):
            assert torch.isfinite(loss)
            losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    return {"seconds": perf_counter() - started, "losses": losses}


@torch.inference_mode()
def evaluate(model, panels):
    model.eval()
    results = {}
    for kind, rows in panels.items():
        fresh = {}
        for d, scene, order, stamp in rows:
            picks = []
            for start in range(0, len(order), 32):
                indices = slice(start, start + 32)
                tokens, key, value = encode(pack(scene, indices), order[indices], hide_query_clock=True)
                predicted = labels(model(tokens))
                picks.extend(predicted.eq(torch.stack((key, value), dim=-1)).tolist())
            fresh[str(d)] = {"fixture_sha256": stamp, "outcomes": picks}
        results[kind] = fresh
    return results


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    training = banks(a.seed)
    holdout = {"original": [], "corrections": []}
    for d in (16, 48, 128, 512):
        raw = generate(a.seed + 761419 + d, 512, d, "mixed", "cuda")
        order = record_order(len(raw[0]), d, a.seed + 64013 + d, "cuda")
        changed, _ = intervention(raw)
        for name, scene in zip(holdout, (raw, changed), strict=True):
            holdout[name].append((d, scene, order, fingerprint(scene, order)))
    archive = {k: archive_panels(json.loads(getattr(a, k).read_text()), a.seed) for k in ("earlier", "later")}
    variants = COMPACTION if a.test_compaction else VARIANTS
    schemes = tuple((g, mode) for g in ("dot", "radial") for mode in variants)
    outputs = {}
    for geometry, mode in (*schemes[a.order:], *schemes[:a.order]):
        decoder = make_decoder(f"{geometry}_without_position", a.seed, "cuda")
        net = ClockDecoder(decoder, mode)
        stamp = state_sha(decoder, common=True)
        fitted = fit(net, training)
        measured = evaluate(net, holdout)
        native = {k: read_archive(net, "eager_fp32", ar) for k, ar in archive.items()}
        outputs[f"{geometry}/{mode}"] = {"initial_sha256": stamp, "fit": fitted, "procedural": measured, "archive": native}
        print(geometry, mode, fitted["seconds"], flush=True)
    assert len({p["initial_sha256"] for p in outputs.values()}) == 1
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "steps": STEPS, "schedule": SCHEDULE,
            "banks": {d: v[-1] for d, v in training.items()},
            "archive_sha256": {k: digest(getattr(a, k)) for k in archive}, "results": outputs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("earlier", "later", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--order", type=int, choices=range(6), required=True)
    p.add_argument("--test-compaction", action="store_true")
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
