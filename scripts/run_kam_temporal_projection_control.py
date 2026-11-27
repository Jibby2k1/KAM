#!/usr/bin/env python3
"""Separate spherical scoring from added linear and nonlinear Q/K maps in temporal memory."""

import argparse
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from random import Random

import torch
from torch import nn

from kam.attention import PairwiseAttentionScore
from kam.transformer.retrieval_attention import ResidualFeatures
from run_kam_bitemporal_decoder import encode, make_decoder, pack
from run_kam_clock_release_training import ENCODING, source_panels
from run_kam_radial_archive_training import assess
from run_kam_radial_curriculum_order import state_sha
from run_kam_temporal_clock_encoding import ClockDecoder, SCHEDULE, banks, evaluate, exposure_order, fit, holdout_panels

POLICIES = ("dot", "spherical", "spherical_linear", "spherical_nonlinear")
BOTTLENECK = 16


class FixedSpherical(PairwiseAttentionScore):
    def forward(self, queries, keys):
        return sqrt(self.head_dim) * super().forward(queries, keys)


class ResidualQK(nn.Module):
    def __init__(self, base, width, bottleneck, seed, nonlinear):
        super().__init__()
        self.base = base
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.query = ResidualFeatures(width, bottleneck)
            self.key = ResidualFeatures(width, bottleneck)
        if not nonlinear:
            self.query.activation = self.key.activation = nn.Identity()

    def forward(self, inputs):
        query, key, value = self.base(inputs).chunk(3, dim=-1)
        return torch.cat((query + self.query(inputs), key + self.key(inputs), value), dim=-1)


def decoder(policy, seed):
    model = make_decoder("dot_without_position", seed, "cpu")
    initial_common = state_sha(model)
    if policy != "dot":
        for i, block in enumerate(model.blocks):
            context = block.context
            context.score = FixedSpherical(context.num_heads, context.head_dim, score_type="cosine")
            if policy != "spherical":
                context.qkv = ResidualQK(context.qkv, context.d_model, BOTTLENECK, seed + 140918 + i,
                                         nonlinear=policy == "spherical_nonlinear")
    return model.to("cuda"), initial_common


@torch.inference_mode()
def zero_residual_test(models, data):
    reference = models["spherical"](data)
    return all(torch.equal(reference, models[p](data)) for p in POLICIES[-2:])


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    baseline = json.loads(a.reference.read_bytes())
    assert baseline["seed"] == a.seed and baseline["training_domain"] == "mixed"
    training = banks(a.seed, "mixed", SCHEDULE)
    assert {str(d): p[-1] for d, p in training.items()} == baseline["banks"]
    staged, archive = source_panels(a)
    check = {k: sha256(f.read_bytes()).hexdigest() for k, f in staged.items()}
    assert baseline["archive_sha256"]["external"] == check["unemployment"]
    held = holdout_panels(a.seed, (16, 48, 128, 512))
    order = exposure_order(SCHEDULE, a.seed, False)
    models = {p: decoder(p, a.seed) for p in POLICIES}
    hashes = {stamp for _, stamp in models.values()}
    assert len(hashes) == 1
    assert hashes.pop() == baseline["results"]["dot/" + ENCODING]["initial_sha256"]
    active = {p: ClockDecoder(net, ENCODING) for p, (net, _) in models.items()}
    mk = slice(0, 24)
    sample = holdout_panels(a.seed, (16,))["original"][0]
    audit_inputs, _, _ = encode(pack(sample[1], mk), sample[2][mk], hide_query_clock=True)
    assert zero_residual_test(active, audit_inputs)
    policy_stats = {}
    for p in Random(a.seed + 98819).sample(list(POLICIES), len(POLICIES)):
        model = active[p]
        training_result = fit(model, training, order)
        evaluated = {k: assess(model, "eager_fp32", panel) for k, panel in archive.items()}
        simulated = evaluate(model, held)
        if p == "dot":
            old = baseline["results"]["dot/" + ENCODING]
            assert training_result["losses"] == old["fit"]["losses"]
            assert simulated == old["procedural"]
            assert evaluated["unemployment"] == old["archive"]["external"]
        policy_stats[p] = {"train": training_result, "heldout": simulated, "archives": evaluated,
                           "parameters": sum(v.numel() for v in model.parameters())}
        print(p, training_result["seconds"], flush=True)
    assert policy_stats["dot"]["parameters"] == policy_stats["spherical"]["parameters"]
    assert policy_stats["spherical_linear"]["parameters"] == policy_stats["spherical_nonlinear"]["parameters"]
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "clock": ENCODING, "zero_residual_parity": True,
            "steps": len(order), "tokens": baseline["train_tokens"], "bottleneck": BOTTLENECK,
            "sources": check, "reference": sha256(a.reference.read_bytes()).hexdigest(), "arms": policy_stats}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("reference", "dallas", "unemployment", "industry", "later-unemployment", "later-industry", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
