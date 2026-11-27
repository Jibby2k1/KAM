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
from kam_archive_projection import generated_archive_scene
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


def banks(seed, domain):
    by_depth = {}
    for d, count in SCHEDULE:
        scene_seed = seed + 681133 + d
        revision, archive, changed = None, None, None
        if domain != "archive_shaped":
            revision, changed = intervention(generate(scene_seed, 4096, d, "mixed", "cuda"))
        if domain != "corrections":
            archive = generated_archive_scene(scene_seed, 4096, d, "cuda")
        scenes = tuple(s for s in (revision, archive) if s is not None)
        order = record_order(4096, d, seed + 772709 + d, "cuda")
        g = torch.Generator().manual_seed(seed + 575019 + d)
        sample = torch.randint(4096, (STEPS // 2, count), generator=g)
        fingerprints = [fingerprint(scene, order) for scene in scenes]
        by_depth[d] = scenes, order, sample.cuda(), {
            "scene_sha256": fingerprints[0] if len(scenes) == 1 else fingerprints,
            "sample_sha256": sha256(sample.numpy().tobytes()).hexdigest(),
            "records_changed": changed,
        }
    return by_depth


def fit(model, panels):
    opt = torch.optim.AdamW(model.parameters(), lr=.002, fused=True)
    model.train()
    torch.cuda.synchronize()
    started = perf_counter()
    losses = []
    for k in range(STEPS):
        d, _ = SCHEDULE[k % len(SCHEDULE)]
        scenes, order, batches, _ = panels[d]
        slots = batches[k // len(SCHEDULE)]
        assert len(slots) % len(scenes) == 0
        blocks = [encode(pack(scene, indices), order[indices], hide_query_clock=True)
                  for scene, indices in zip(scenes, slots.chunk(len(scenes)), strict=True)]
        if len(blocks) == 1:
            tokens, key, value = blocks[0]
        else:
            tokens, key, value = (torch.cat(parts) for parts in zip(*blocks, strict=True))
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
    domain = "mixed" if a.mixed_training else ("archive_shaped" if a.archive_shaped else "corrections")
    training = banks(a.seed, domain)
    holdout = {"original": [], "corrections": []}
    for d in (16, 48, 128, 512):
        raw = generate(a.seed + 761419 + d, 512, d, "mixed", "cuda")
        order = record_order(len(raw[0]), d, a.seed + 64013 + d, "cuda")
        changed, _ = intervention(raw)
        for name, scene in zip(holdout, (raw, changed), strict=True):
            holdout[name].append((d, scene, order, fingerprint(scene, order)))
    sources = {k: getattr(a, k) for k in ("earlier", "later")}
    if a.external:
        sources["external"] = a.external
    archive = {k: archive_panels(json.loads(file.read_text()), a.seed) for k, file in sources.items()}
    if a.external:
        def rescale(panel):
            tokens, entity, value = panel["data"]
            tokens = tokens.clone()
            tokens[..., -1].mul_(3)
            return panel | {"data": (tokens, entity, value)}

        archive["external_clock_times_3"] = [rescale(panel) for panel in archive["external"]]
        variants = ("linear", "signed_log")
    else:
        variants = (*VARIANTS, "signed_log_compact") if domain != "corrections" else (COMPACTION if a.test_compaction else VARIANTS)
    schemes = tuple((g, mode) for g in ("dot", "radial") for mode in variants)
    outputs = {}
    start = a.order % len(schemes)
    for geometry, mode in (*schemes[start:], *schemes[:start]):
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
            "training_domain": domain,
            "banks": {d: v[-1] for d, v in training.items()},
            "archive_sha256": {k: digest(file) for k, file in sources.items()}, "results": outputs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("earlier", "later", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--order", type=int, choices=range(6), required=True)
    p.add_argument("--external", type=Path)
    variants = p.add_mutually_exclusive_group()
    variants.add_argument("--test-compaction", action="store_true")
    variants.add_argument("--archive-shaped", action="store_true")
    variants.add_argument("--mixed-training", action="store_true")
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
