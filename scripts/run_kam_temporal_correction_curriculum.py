#!/usr/bin/env python3
"""Isolate the effect of same-date corrections at matched model and token budgets."""

import argparse
import json
from pathlib import Path

import torch

from audit_kam_natural_data import digest
from run_kam_bitemporal_decoder import fingerprint, make_decoder, record_order
from run_kam_bitemporal_noise import exact, generate
from run_kam_radial_archive_training import archive_panels, assess as assess_archive
from run_kam_radial_curriculum_order import CONFIG, fit, setup, state_sha, training_order

ARMS = tuple((geometry, timing) for geometry in ("dot", "radial") for timing in ("original", "corrections"))
DEPTHS = (48, 128, 512, 1024)
STEPS = 9600


def correction_clocks(bank, anchor, query, identity):
    row = torch.arange(len(bank), device=bank.device)
    target = bank[row, anchor]
    same_key = (bank[..., 0] == target[:, None, 0]) & (bank[..., identity] == target[:, None, identity])
    effective, reported = bank[..., 3], bank[..., 4]
    stale = same_key & (effective < target[:, None, 3]) & (reported <= query[:, None, 3])
    late = same_key & (effective <= query[:, None, 2]) & (reported > query[:, None, 3])
    changed = bank.clone()
    changed[..., 3] = torch.where(stale | late, target[:, None, 3], effective)
    changed[..., 4] = torch.where(stale, torch.minimum(reported, target[:, None, 4] - 1), reported)
    assert torch.equal(changed[..., :3], bank[..., :3])
    assert torch.equal(changed[row, anchor], target)
    magnitude = int(changed[..., 3:].ne(bank[..., 3:]).any(-1).sum())
    return changed, magnitude


def intervention(scene):
    query, alias, fact, a_target, f_target, value = scene
    changed_alias, ca = correction_clocks(alias, a_target, query, 1)
    changed_fact, cf = correction_clocks(fact, f_target, query, 1)
    a, f = exact(query, changed_alias, changed_fact, "both")
    assert torch.equal(a, a_target) and torch.equal(f, f_target) and ca > 0 and cf > 0
    return (query, changed_alias, changed_fact, a_target, f_target, value), {"alias": ca, "fact": cf}


def panels(seed):
    held = {}
    for depth in DEPTHS:
        scene = generate(seed + 652051 + depth, 384, depth, "mixed", "cuda")
        order = record_order(len(scene[0]), depth, seed + 245137 + depth, "cuda")
        shifted, _ = intervention(scene)
        held[depth] = (scene, order, fingerprint(scene, order)), (shifted, order, fingerprint(shifted, order))
    return held


def run(a):
    from run_kam_radial_dispatch import assess

    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    assert len(CONFIG) == 4
    history, generated = setup(a.seed, STEPS)
    study = panels(a.seed)
    shifted, modified = {}, {}
    for d, (scene, order, batches, stamp) in history.items():
        scenes, modified[d] = intervention(scene)
        shifted[d] = scenes, order, batches, (fingerprint(scenes, order), stamp[1])
    ordered = training_order(STEPS, "short_first", a.seed)
    archive = {k: json.loads(getattr(a, k).read_text()) for k in ("earlier", "later")}
    tests = [{r["id"] for r in f["rows"]["test"]} for f in archive.values()]
    assert tests[0].isdisjoint(tests[1])
    archive = {k: archive_panels(fixture, a.seed) for k, fixture in archive.items()}
    outcomes = {}
    for geometry, timing in (*ARMS[a.order:], *ARMS[:a.order]):
        name = f"{geometry}_without_position"
        model, reference = (make_decoder(name, a.seed, "cuda") for _ in range(2))
        stamp = state_sha(model, common=True)
        fitted = fit(model, reference, history if timing == "original" else shifted, generated, ordered, "eager")
        outputs = {label: assess(reference, "eager_fp32", {d: items[i] for d, items in study.items()})
                   for i, label in enumerate(("original", "corrections"))}
        archives = {k: assess_archive(reference, "eager_fp32", p) for k, p in archive.items()}
        outcomes[f"{geometry}/{timing}"] = {"initial_shared_sha256": stamp, "training": fitted,
                                           "synthetic": outputs, "archive": archives}
    assert len({r["initial_shared_sha256"] for r in outcomes.values()}) == 1
    assert len({r["training"][str(STEPS)]["tokens"] for r in outcomes.values()}) == 1
    return {"device": torch.cuda.get_device_name(), "seed": a.seed, "training_steps": STEPS, "train_order": "short_first",
            "source_sha256": {k: digest(getattr(a, k)) for k in archive},
            "baseline_banks": {str(k): bank[-1] for k, bank in history.items()},
            "shifted_banks": {str(k): bank[-1] for k, bank in shifted.items()},
            "changed_records": modified, "outcomes": outcomes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("earlier", "later", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--order", type=int, choices=range(len(ARMS)), required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
