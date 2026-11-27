#!/usr/bin/env python3
"""Train off-release cutoffs on matched synthetic batches and replay untouched archives."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import torch

from run_kam_bitemporal_decoder import encode, make_decoder, pack, record_order
from run_kam_radial_archive_training import archive_panels, assess
from run_kam_radial_curriculum_order import state_sha
from run_kam_temporal_clock_encoding import ClockDecoder, SCHEDULE, banks, evaluate, exposure_order, fit, holdout_panels

PRECISION = 64
VARIANTS = ("reference", "all_fact_clocks", "wrong_address_only")
MIXTURES = {"phase_quarter": 16, "phase_half": 32, "phase_three_quarters": 48}
ENCODING = "log_episode_mean"


def cutoffs(training, seed):
    offsets, sealed = {}, {}
    for d, (scenes, _, _, _) in training.items():
        latest = scenes[-1]
        rng = torch.Generator().manual_seed(seed + 1104259 + d)
        offset = torch.randint(PRECISION, (len(latest[0]),), generator=rng)
        sealed[str(d)] = sha256(offset.numpy().tobytes()).hexdigest()
        gates = torch.randint(PRECISION, (len(latest[0]),), generator=rng)
        base = offset.cuda().float() / (PRECISION * 24)
        gate = gates.cuda()
        offsets[d] = {"all": base} | {m: base * gate.lt(boundary) for m, boundary in MIXTURES.items()}
        sealed[f"{d}/mixture_gate"] = sha256(gates.numpy().tobytes()).hexdigest()
    return offsets, sealed


def shift(tokens, entity, offsets, wrong_only):
    fact = tokens[..., 233].bool()
    admitted = fact
    if wrong_only:
        matched_entity = tokens[..., 40:104].gather(2, entity[:, None, None].expand(-1, tokens.shape[1], 1)).squeeze(2)
        query_namespace = tokens[:, -1, :8].argmax(-1)
        same_namespace = tokens[..., :8].gather(2, query_namespace[:, None, None].expand(-1, tokens.shape[1], 1)).squeeze(2)
        admitted = fact & ~(matched_entity.bool() & same_namespace.bool())
    tokens[..., -1] -= offsets[:, None] * admitted
    return tokens


def transform(mode, shifts):
    if mode == "reference":
        return None

    def apply(tokens, entity, d, slots, groups):
        count = len(slots) // groups
        plan = shifts[d][mode if mode in MIXTURES else "all"]
        shift(tokens[-count:], entity[-count:], plan[slots[-count:]], mode == "wrong_address_only")
        return tokens
    return apply


def check_fixtures(training, shifts):
    for d, (scenes, _, _, _) in training.items():
        scene = scenes[-1]
        n = len(scene[0])
        # The integer fixture has no publication at a fractional step. A shift
        # smaller than one step cannot change admission or same-address order.
        baseline = shifts[d]["all"]
        assert bool((baseline >= 0).all() and (baseline < 1/24).all())
        assert all(torch.logical_or(v == 0, v == baseline).all() for v in shifts[d].values())
        span = slice(0, min(n, 96))
        order = record_order(min(n, 96), d, 618441 + d, scene[0].device)
        raw, key, _ = encode(pack(scene, span), order, hide_query_clock=True)
        for sham in (False, True):
            perturbed = shift(raw.clone(), key, baseline[span], sham)
            assert torch.equal(raw[..., -1].le(0), perturbed[..., -1].le(0))
            assert torch.equal(raw[..., :-1], perturbed[..., :-1])
            changed = perturbed[..., -1].ne(raw[..., -1])
            assert bool(changed.any() and not changed[:, -1].any())


def source_panels(a):
    inputs = {k: getattr(a, k) for k in ("dallas", "unemployment", "industry", "later_unemployment", "later_industry")
              if getattr(a, k)}
    outputs = {}
    for name, file in inputs.items():
        panels = archive_panels(json.loads(file.read_bytes()), a.seed)
        if name != "dallas":
            for panel in panels:
                panel["data"][0][..., -1].mul_(1/30)
        outputs[name] = panels
    return inputs, outputs


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    ref = json.loads(a.reference.read_bytes())
    assert ref["seed"] == a.seed and ref["training_domain"] == a.domain and ref["external_base_units"] == 30
    training = banks(a.seed, a.domain, SCHEDULE)
    assert {str(d): p[-1] for d, p in training.items()} == ref["banks"]
    offsets, seal = cutoffs(training, a.seed)
    check_fixtures(training, offsets)
    paths, archive = source_panels(a)
    hashes = {k: sha256(v.read_bytes()).hexdigest() for k, v in paths.items()}
    assert ref["archive_sha256"]["external"] == hashes["unemployment"]
    update = exposure_order(SCHEDULE, a.seed, False)
    test = holdout_panels(a.seed, (16, 48, 128, 512))
    for case, depths in test.items():
        for depth, _, _, hashed in depths:
            assert hashed == ref["results"]["radial/" + ENCODING]["procedural"][case][str(depth)]["fixture_sha256"]
    policies = ("reference", *MIXTURES) if a.compare_mixtures else VARIANTS
    arms = [(g, t) for g in ("dot", "radial") for t in policies]
    output = {}
    audit = {}
    for geometry, arm in Random(a.seed + 831557).sample(arms, len(arms)):
        model = make_decoder(geometry + "_without_position", a.seed, "cuda")
        archive_control = ref["results"][geometry + "/" + ENCODING]
        assert state_sha(model, common=True) == archive_control["initial_sha256"]
        net = ClockDecoder(model, ENCODING)
        elapsed = fit(net, training, update, transform=transform(arm, offsets))
        synthetic = evaluate(net, test)
        result = {k: assess(net, "eager_fp32", panels) for k, panels in archive.items()}
        if arm == "reference":
            assert result["unemployment"] == archive_control["archive"]["external"]
            stored = archive_control["procedural"]
            replay = {}
            for case, depths in synthetic.items():
                for depth, got in depths.items():
                    old = stored[case][depth]
                    assert old["fixture_sha256"] == got["fixture_sha256"]
                    replay[f"{case}/{depth}"] = sum(a == b for a, b in zip(old["outcomes"], got["outcomes"], strict=True))
            loss_delta = max(abs(a - b) for a, b in zip(elapsed["losses"], archive_control["fit"]["losses"], strict=True))
            assert loss_delta == 0 and set(replay.values()) == {512}
            audit[geometry] = {"unemployment_replay": True, "procedural_outcomes_replayed": replay,
                               "max_loss_difference": loss_delta}
            print(geometry, "reference replay", audit[geometry], flush=True)
        output[f"{geometry}/{arm}"] = {"fit": elapsed, "archives": result, "procedural": synthetic}
        print(geometry, arm, elapsed["seconds"], flush=True)
    return {"seed": a.seed, "training_domain": a.domain, "reference_sha256": sha256(a.reference.read_bytes()).hexdigest(),
            "device": torch.cuda.get_device_name(), "steps": ref["steps"], "tokens": ref["train_tokens"],
            "schedule": ref["schedule"], "fixture_hashes": hashes, "training_banks": ref["banks"],
            "phase_offset_sha256": seal, "clock": ENCODING, "arms": output, "reference_audit": audit,
            "phase_mixture_fractions": {k: v/PRECISION for k, v in MIXTURES.items()} if a.compare_mixtures else {}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--domain", choices=("mixed", "archive_shaped"), required=True)
    p.add_argument("--compare-mixtures", action="store_true")
    for name in ("reference", "dallas", "unemployment", "industry", "later_unemployment", "later_industry", "output"):
        p.add_argument("--" + name.replace("_", "-"), required=name in ("reference", "dallas", "unemployment", "output"), type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
