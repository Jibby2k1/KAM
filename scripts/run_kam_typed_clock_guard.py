#!/usr/bin/env python3
"""Cross trusted clock admission with within-address rank in position-free decoding."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import torch

from kam_address_local_clock import local_features
from run_kam_bitemporal_decoder import encode, make_decoder, pack
from run_kam_clock_release_training import source_panels
from run_kam_radial_archive_training import assess
from run_kam_radial_curriculum_order import state_sha
from run_kam_temporal_clock_encoding import SCHEDULE, banks, evaluate, exposure_order, fit, holdout_panels, recode

ENCODING = "log_available_mean"
POLICIES = ("timestamps", "timestamps_and_rank", "hard_clock", "hard_clock_and_rank")


def admit_order(raw):
    observed = raw[..., -2:].le(0).all(-1)
    index = (~observed).long().argsort(dim=1, stable=True)
    q = observed.sum(1) - 1
    return index, q


def guarded_tokens(raw, *, ranked):
    tokens = recode(raw, ENCODING)
    if ranked:
        tokens = local_features(raw, tokens, alias=False, sham=False)
    order, query = admit_order(raw)
    return tokens.gather(1, order[..., None].expand_as(tokens)), query


class GuardDecoder(torch.nn.Module):
    def __init__(self, decoder, policy):
        super().__init__()
        self.decoder = decoder
        self.ranks = policy.endswith("rank")
        self.guard = policy.startswith("hard_clock")

    def forward(self, raw):
        if not self.guard:
            tokens = recode(raw, ENCODING)
            if self.ranks:
                tokens = local_features(raw, tokens, alias=False, sham=False)
            return self.decoder(tokens)
        arranged, query = guarded_tokens(raw, ranked=self.ranks)
        hidden, _ = self.decoder.encode(arranged)
        at_query = hidden[torch.arange(len(hidden), device=hidden.device), query]
        return self.decoder.readout(at_query)


@torch.inference_mode()
def verify(panels, model):
    d, scene, order, _ = panels["corrections"][1]
    batch = slice(0, 32)
    raw, _, _ = encode(pack(scene, batch), order[batch], hide_query_clock=True)
    indices, query = admit_order(raw)
    assert (indices.gather(1, query[:, None]) == raw.shape[1]-1).all()
    observed = raw[..., -2:].le(0).all(-1)
    kept = torch.arange(raw.shape[1], device=raw.device)[None] <= query[:, None]
    assert torch.equal(observed.gather(1, indices), kept)
    modified = raw.clone()
    # Change only payloads and addresses of the unavailable records.
    modified[..., :232] = torch.where(observed[..., None], raw[..., :232], raw[..., :232].roll(1, dims=-1))
    before = model(raw)
    after = model(modified)
    assert torch.equal(before, after)
    return {"depth": d, "exact_hidden_suffix_null_effect": len(raw), "excluded": int((~observed).sum())}


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    reference = json.loads(a.reference.read_bytes())
    assert reference["seed"] == a.seed and reference["training_domain"] == "mixed"
    plans = banks(a.seed, "mixed", SCHEDULE)
    assert {str(k): v[-1] for k, v in plans.items()} == reference["banks"]
    order = exposure_order(SCHEDULE, a.seed, False)
    held = holdout_panels(a.seed, (16, 48, 128, 512, 1024))
    files, targets = source_panels(a)
    identities = {k: sha256(p.read_bytes()).hexdigest() for k, p in files.items()}
    effects = {}
    design = [(g, p) for g in ("dot", "radial") for p in POLICIES]
    for geometry, policy in Random(a.seed+998102).sample(design, len(design)):
        model = make_decoder(f"{geometry}_without_position", a.seed, "cuda")
        assert state_sha(model, common=True) == reference["results"][f"{geometry}/log_episode_mean"]["initial_sha256"]
        net = GuardDecoder(model, policy)
        checked = verify(held, net) if net.guard else {}
        trained = fit(net, plans, order)
        synthetic = evaluate(net, held)
        target = {k: assess(net, "eager_fp32", records) for k, records in targets.items()}
        effects[f"{geometry}/{policy}"] = {"fit": trained, "generated": synthetic, "archives": target, "audit": checked}
        print(geometry, policy, trained["seconds"], flush=True)
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "clock": ENCODING,
            "steps": len(order), "token_count": reference["train_tokens"], "sources": identities, "arms": effects}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("reference", "dallas", "unemployment", "industry", "later-unemployment", "later-industry", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
