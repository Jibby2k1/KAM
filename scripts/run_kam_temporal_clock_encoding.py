#!/usr/bin/env python3
"""Test clock feature scaling and an explicit eligibility guard on correction histories."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import cos, log1p, pi
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from benchmark_kam_radial_backward import objective
from benchmark_kam_radial_sdpa import labels
from kam_archive_projection import generated_archive_scene
from run_kam_bitemporal_decoder import encode, fingerprint, make_decoder, pack, record_order
from run_kam_bitemporal_noise import exact, generate, scalar_oracle
from run_kam_radial_archive_training import archive_panels, assess as read_archive
from run_kam_radial_curriculum_order import state_sha
from run_kam_temporal_correction_curriculum import intervention

VARIANTS = ("linear", "signed_log", "signed_log_guard")
COMPACTION = ("signed_log", "signed_log_guard", "signed_log_compact")
OUTLIER_VARIANTS = ("signed_log", "log_episode_mean", "log_episode_median", "log_available_mean")
SCHEDULE = ((16, 100), (48, 34))
LONG_SCHEDULE = ((16, 82), (48, 42), (128, 13), (512, 3))
assert 2 * sum((2*d+1)*n for d, n in SCHEDULE) == sum((2*d+1)*n for d, n in LONG_SCHEDULE)
STEPS = 3200


def recode(tokens, mode):
    if mode == "linear":
        return tokens
    clock = tokens[..., -2:]
    if mode == "ordinal":
        ordered, slots = torch.sort(clock, dim=1, stable=True)
        distinct = torch.cat((torch.zeros_like(ordered[:, :1], dtype=torch.bool),
                              ordered[:, 1:].ne(ordered[:, :-1])), dim=1)
        ranks = distinct.cumsum(dim=1)
        reversed_ranks = torch.zeros_like(ranks).scatter_(1, slots, ranks)
        centered = reversed_ranks - reversed_ranks[:, -1:]
        values = centered / ranks[:, -1:].clamp_min(1)
    elif mode in ("episode_scale", "log_episode_max", "log_episode_mean", "log_episode_median", "log_available_mean"):
        spread = clock.abs()
        if mode in ("log_episode_mean", "log_available_mean"):
            held = spread if mode == "log_episode_mean" else spread.masked_fill(~clock.le(0).all(-1, keepdim=True), 0)
            scale = held.sum(dim=1, keepdim=True) / held.gt(0).sum(dim=1, keepdim=True).clamp_min(1)
        elif mode == "log_episode_median":
            sorted_spread = spread.masked_fill(spread.eq(0), torch.inf).sort(dim=1).values
            median = (spread.gt(0).sum(dim=1, keepdim=True).clamp_min(1) - 1) // 2
            scale = sorted_spread.gather(1, median)
        else:
            scale = spread.amax(dim=1, keepdim=True)
        values = clock / scale.clamp_min(1e-12)
        if mode != "episode_scale":
            values = values.sign() * (24 * values.abs()).log1p() / log1p(24)
    else:
        values = clock.sign() * (clock.abs() * 24).log1p() / log1p(24)
    output = torch.cat((tokens[..., :-2], values), dim=-1)
    if mode == "signed_log_guard":
        available = clock.le(0).all(-1, keepdim=True)
        output = output.masked_fill(~available, 0)
    return output


def compact_tokens(tokens):
    # The final query is the last eligible token. Move rejected events after
    # it so causal attention cannot see them at any layer.
    eligible = tokens[..., -2:].le(0).all(-1)
    order = (~eligible).to(torch.int32).argsort(dim=1, stable=True)
    data = recode(tokens, "signed_log").gather(1, order.unsqueeze(-1).expand_as(tokens))
    return data, eligible.sum(dim=1) - 1


def displaced_publication(scene, seed, kind):
    assert kind in ("future_outlier", "old_wrong_entity_outlier")
    query, aliases, facts, gold_alias, gold_fact, gold_value = scene
    known = facts[..., 4] - query[:, 3:4]
    if kind == "future_outlier":
        permitted = known.gt(0)
    else:
        entity = aliases[..., 2].gather(1, gold_alias[:, None])
        permitted = known.lt(0) & facts[..., 1].ne(entity)
    generator = torch.Generator().manual_seed(seed + 892663)
    choices = torch.rand(facts.shape[:2], generator=generator).masked_fill(~permitted, -1).argmax(dim=-1)
    rows = permitted.any(dim=-1).nonzero().flatten()
    picked = choices[rows]
    changed = facts.clone()
    changed[rows, picked, 4] = query[rows, 3] + 64 * known[rows, picked]
    assert torch.equal((changed[..., 4] - query[:, 3:4]).sign(), known.sign())
    assert int(changed.ne(facts).sum()) == len(rows)
    recovered = exact(query, aliases, changed, "both")
    assert all(torch.equal(a, b) for a, b in zip(recovered, (gold_alias, gold_fact), strict=True))
    oracle = scalar_oracle(query, aliases, changed)
    assert oracle == (gold_alias.tolist(), gold_fact.tolist())
    return (query, aliases, changed, gold_alias, gold_fact, gold_value), permitted.any(dim=-1).tolist()


def snap_release_phase(scene, seed, target_only):
    del seed
    query, aliases, facts, gold_alias, gold_fact, gold_value = scene
    latest = facts[..., 4].gather(1, gold_fact[:, None])
    gap = query[:, 3:4] - latest
    assert gap.ge(0).all()
    address = aliases[..., 2].gather(1, gold_alias[:, None])
    same = facts[..., 1].eq(address) & facts[..., 0].eq(query[:, :1])
    changed = facts.clone()
    changed[..., 4] += gap * (same if target_only else 1)
    assert (changed[..., 4].gather(1, gold_fact[:, None]) == query[:, 3:4]).all()
    was_visible = facts[..., 4] <= query[:, 3:4]
    now_visible = changed[..., 4] <= query[:, 3:4]
    assert (was_visible[same] == now_visible[same]).all()
    if target_only:
        assert torch.equal(changed[~same], facts[~same])
    a, b = exact(query, aliases, changed, "both")
    assert torch.equal(a, gold_alias) and torch.equal(b, gold_fact)
    assert scalar_oracle(query, aliases, changed) == (gold_alias.tolist(), gold_fact.tolist())
    return (query, aliases, changed, gold_alias, gold_fact, gold_value), gap.squeeze(-1).tolist()


class ClockDecoder(torch.nn.Module):
    def __init__(self, decoder, mode):
        super().__init__()
        self.decoder = decoder
        self.mode = mode

    def forward(self, tokens):
        if self.mode != "signed_log_compact":
            return self.decoder(recode(tokens, self.mode))
        data, query = compact_tokens(tokens)
        hidden, _ = self.decoder.encode(data)
        return self.decoder.readout(hidden[torch.arange(len(hidden), device=hidden.device), query])


def banks(seed, domain, schedule):
    by_depth = {}
    for d, count in schedule:
        scene_seed = seed + 681133 + d
        total = min(4096, 2**20 // d)
        revision, archive, changed = None, None, None
        if domain != "archive_shaped" or d > 48:
            revision, changed = intervention(generate(scene_seed, total, d, "mixed", "cuda"))
        if domain != "corrections" and d <= 48:
            archive = generated_archive_scene(scene_seed, total, d, "cuda")
        scenes = tuple(s for s in (revision, archive) if s is not None)
        order = record_order(total, d, seed + 772709 + d, "cuda")
        g = torch.Generator().manual_seed(seed + 575019 + d)
        sample = torch.randint(total, (STEPS // len(schedule), count), generator=g)
        fingerprints = [fingerprint(scene, order) for scene in scenes]
        by_depth[d] = scenes, order, sample.cuda(), {
            "scene_sha256": fingerprints[0] if len(scenes) == 1 else fingerprints,
            "sample_sha256": sha256(sample.numpy().tobytes()).hexdigest(),
            "records_changed": changed,
        }
    return by_depth


def exposure_order(schedule, seed, delay):
    complete = [d for d, _ in schedule] * (STEPS // len(schedule))
    if not delay:
        return complete
    prefix = [16, 48] * 400
    counts = Counter(complete)
    counts.subtract(prefix)
    tail = list(counts.elements())
    Random(seed + 113281).shuffle(tail)
    result = prefix + tail
    assert Counter(result) == Counter(complete)
    return result


def fit(model, panels, order, *, transform=None):
    opt = torch.optim.AdamW(model.parameters(), lr=.002, fused=True)
    model.train()
    torch.cuda.synchronize()
    started = perf_counter()
    losses = []
    cursors = Counter()
    for step, d in enumerate(order, 1):
        scenes, permutation, batches, _ = panels[d]
        slots = batches[cursors[d]]
        cursors[d] += 1
        assert len(slots) % len(scenes) == 0
        blocks = [encode(pack(scene, indices), permutation[indices], hide_query_clock=True)
                  for scene, indices in zip(scenes, slots.chunk(len(scenes)), strict=True)]
        if len(blocks) == 1:
            tokens, key, value = blocks[0]
        else:
            tokens, key, value = (torch.cat(parts) for parts in zip(*blocks, strict=True))
        if transform:
            tokens = transform(tokens, key, d, slots, len(scenes))
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
    assert all(cursors[d] == len(v[2]) for d, v in panels.items())
    return {"seconds": perf_counter() - started, "losses": losses}


@torch.inference_mode()
def evaluate(model, panels):
    model.eval()
    results = {}
    for kind, rows in panels.items():
        fresh = {}
        for d, scene, order, stamp in rows:
            picks = []
            batch = 4 if d == 1024 else 32
            for start in range(0, len(order), batch):
                indices = slice(start, start + batch)
                tokens, key, value = encode(pack(scene, indices), order[indices], hide_query_clock=True)
                predicted = labels(model(tokens))
                picks.extend(predicted.eq(torch.stack((key, value), dim=-1)).tolist())
            fresh[str(d)] = {"fixture_sha256": stamp, "outcomes": picks}
        results[kind] = fresh
    return results


def holdout_panels(seed, lengths, device="cuda"):
    holdout = {"original": [], "corrections": []}
    for d in lengths:
        raw = generate(seed + 761419 + d, 512, d, "mixed", device)
        order = record_order(len(raw[0]), d, seed + 64013 + d, device)
        changed, _ = intervention(raw)
        for name, scene in zip(holdout, (raw, changed), strict=True):
            holdout[name].append((d, scene, order, fingerprint(scene, order)))
    return holdout


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    domain = "mixed" if a.mixed_training else ("archive_shaped" if a.archive_shaped else "corrections")
    assert not a.long_exposure or domain != "archive_shaped"
    schedule = LONG_SCHEDULE if a.long_exposure else SCHEDULE
    training = banks(a.seed, domain, schedule)
    update_order = exposure_order(schedule, a.seed, a.short_delay)
    lengths = (16, 48, 128, 512, 1024) if a.long_exposure or a.long_probe else (16, 48, 128, 512)
    holdout = holdout_panels(a.seed, lengths)
    sources = {k: getattr(a, k) for k in ("earlier", "later")}
    if a.external:
        sources["external"] = a.external
    archive = {k: archive_panels(json.loads(file.read_text()), a.seed) for k, file in sources.items()}
    clock_parity = {}
    if a.external:
        def rescale(panel, factor):
            tokens, entity, value = panel["data"]
            tokens = tokens.clone()
            tokens[..., -1].mul_(factor)
            return panel | {"data": (tokens, entity, value)}

        if a.external_base_units != 1:
            archive["external_raw_units"] = archive["external"]
            archive["external"] = [rescale(panel, 1 / a.external_base_units) for panel in archive["external"]]
        archive["external_clock_times_3"] = [rescale(panel, 3) for panel in archive["external"]]
        variants = ("linear", "signed_log", "episode_scale", "ordinal") if a.ordinal_clock else ("linear", "signed_log")
        if a.relative_log:
            variants = ("linear", "signed_log", "log_episode_max", "log_episode_mean")
        if a.outlier_clock:
            from functools import partial

            variants = OUTLIER_VARIANTS
            for kind in ("future_outlier", "old_wrong_entity_outlier"):
                archive[kind] = archive_panels(json.loads(a.external.read_text()), a.seed,
                                               intervention=partial(displaced_publication, kind=kind))
        if a.release_phase_probe:
            from functools import partial

            for label, target_only in (("target_release_snapped", True), ("all_releases_snapped", False)):
                changed = archive_panels(json.loads(a.external.read_text()), a.seed,
                                         intervention=partial(snap_release_phase, target_only=target_only))
                archive[label] = [rescale(panel, 1 / a.external_base_units) for panel in changed]
        if a.ordinal_clock or a.relative_log:
            probes = zip(archive["external"], archive["external_clock_times_3"], strict=True)
            errors = {mode: [] for mode in variants}
            for before, after in probes:
                for mode in variants:
                    delta = (recode(before["data"][0], mode)[..., -2:] - recode(after["data"][0], mode)[..., -2:]).abs().max()
                    errors[mode].append(float(delta))
            clock_parity = {mode: max(values) for mode, values in errors.items()}
            if a.ordinal_clock:
                assert clock_parity["ordinal"] == 0.0
    elif a.long_exposure or a.long_probe:
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
        fitted = fit(net, training, update_order)
        measured = evaluate(net, holdout)
        native = {k: read_archive(net, "eager_fp32", ar) for k, ar in archive.items()}
        outputs[f"{geometry}/{mode}"] = {"initial_sha256": stamp, "fit": fitted, "procedural": measured, "archive": native}
        print(geometry, mode, fitted["seconds"], flush=True)
    assert len({p["initial_sha256"] for p in outputs.values()}) == 1
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "steps": STEPS, "schedule": schedule,
            "train_tokens": STEPS // len(schedule) * sum((2*d+1)*n for d, n in schedule),
            "training_domain": domain, "short_first_prefix_updates": 800 if a.short_delay else 0,
            "external_base_units": a.external_base_units,
            "order_sha256": sha256(json.dumps(update_order, separators=(",", ":")).encode()).hexdigest(),
            "banks": {d: v[-1] for d, v in training.items()},
            "archive_sha256": {k: digest(file) for k, file in sources.items()}, "results": outputs} | (
                {"external_feature_max_delta": clock_parity} if clock_parity else {}) | (
                {"intervened_rows": {k: [p["changed"] for p in ar] for k, ar in archive.items() if "changed" in ar[0]}}
                if a.outlier_clock or a.release_phase_probe else {})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("earlier", "later", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--order", type=int, choices=range(6), required=True)
    p.add_argument("--external", type=Path)
    p.add_argument("--external-base-units", type=float, default=1.)
    p.add_argument("--long-exposure", action="store_true")
    p.add_argument("--long-probe", action="store_true")
    p.add_argument("--short-delay", action="store_true")
    coordinate = p.add_mutually_exclusive_group()
    coordinate.add_argument("--ordinal-clock", action="store_true")
    coordinate.add_argument("--relative-log", action="store_true")
    coordinate.add_argument("--outlier-clock", action="store_true")
    p.add_argument("--release-phase-probe", action="store_true")
    variants = p.add_mutually_exclusive_group()
    variants.add_argument("--test-compaction", action="store_true")
    variants.add_argument("--archive-shaped", action="store_true")
    variants.add_argument("--mixed-training", action="store_true")
    a = p.parse_args()
    if a.short_delay and not a.long_exposure:
        p.error("delaying long shapes requires the long-exposure schedule")
    if (a.ordinal_clock or a.relative_log or a.outlier_clock) and not a.external:
        p.error("unit-normalized clock comparisons require the external rescaling panel")
    if a.external_base_units <= 0 or (a.external_base_units != 1. and (not a.external or a.outlier_clock)):
        p.error("external publication scale requires a positive conversion and an unperturbed external panel")
    if a.release_phase_probe and not (a.external and a.relative_log):
        p.error("release boundary probes require the relative-log comparison and an external source")
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
