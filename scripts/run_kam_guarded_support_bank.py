#!/usr/bin/env python3
"""Compare static memory supports and a matched pointwise residual in guarded lookup."""

import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from types import MethodType

import torch
from torch import nn

from run_kam_bitemporal_decoder import make_decoder
from run_kam_clock_release_training import source_panels
from run_kam_radial_archive_training import assess
from run_kam_temporal_clock_encoding import SCHEDULE, banks, evaluate, exposure_order, holdout_panels, fit
from run_kam_typed_clock_guard import GuardDecoder


class Pointwise(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(width, 88, bias=False), nn.GELU(), nn.Linear(88, width, bias=False))

    def forward(self, inputs, return_weights=False):
        assert not return_weights
        return self.ffn(inputs), None


def support_read(memory, inputs, return_weights=False, *, k):
    assert not return_weights and memory.support_mask is None and memory.dropout.p == 0
    if k == 0:
        return torch.zeros_like(inputs), None
    queries = memory.project_keys(inputs)
    n, length, _ = inputs.shape
    keys = memory.memory_keys.unsqueeze(0).expand(n, -1, -1, -1)
    logits = memory.score(queries, keys)
    positions = logits.argsort(dim=-1, descending=True, stable=True)[..., :k]
    keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, positions, True)
    weights = logits.masked_fill(~keep, -torch.inf).softmax(-1)
    attended = weights @ memory.memory_values.unsqueeze(0)
    return memory.output(attended.transpose(1, 2).contiguous().view(n, length, memory.d_model)), None


def build(geometry, mode, seed):
    core = make_decoder(f"{geometry}_without_position" if mode == "context" else f"{geometry}_hybrid", seed, "cpu")
    core.no_positions = True
    for block in core.blocks:
        block.context.relative_bias = None
        if mode == "pointwise":
            block.memory = Pointwise(core.d_model)
    return core


def assess_all(model, held, archives):
    return {"generated": evaluate(model, held),
            "archives": {name: assess(model, "eager_fp32", records) for name, records in archives.items()}}


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    training = banks(a.seed, "mixed", SCHEDULE)
    order = exposure_order(SCHEDULE, a.seed, False)
    held = holdout_panels(a.seed, (16, 48, 128, 512, 1024))
    files, archives = source_panels(a)
    prior = json.loads(a.reference.read_bytes())
    assert prior["seed"] == a.seed and prior["device"] == torch.cuda.get_device_name() == "NVIDIA L4"
    common = {g: build(g, "context", a.seed).state_dict() for g in ("dot", "radial")}
    nparams = {}
    results = {}
    plan = [(g, m) for g in common for m in ("context", "support_bank", "pointwise")]
    for g, m in Random(a.seed+70137).sample(plan, len(plan)):
        core = build(g, m, a.seed)
        values = core.state_dict()
        assert all(torch.equal(values[k], v) for k, v in common[g].items())
        label = f"{g}/{m}"
        nparams[label] = sum(p.numel() for p in core.parameters())
        model = GuardDecoder(core.cuda(), "hard_clock_and_rank")
        trained = fit(model, training, order)
        outcome = assess_all(model, held, archives)
        if m == "context":
            expected = prior["arms"][f"{g}/hard_clock_and_rank"]
            assert trained["losses"] == expected["fit"]["losses"]
            assert all(expected[k] == v for k, v in outcome.items())
        results[label] = {"fit": trained, "soft": outcome, "frozen_interventions": {}}
        if m == "support_bank":
            for k in (0, 1, 4):
                for block in core.blocks:
                    block.memory.forward = MethodType(partial(support_read, k=k), block.memory)
                results[label]["frozen_interventions"][str(k)] = assess_all(model, held, archives)
        print(label, trained["seconds"], flush=True)
    for g in common:
        assert abs(nparams[f"{g}/support_bank"] - nparams[f"{g}/pointwise"]) <= 136
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "steps": len(order), "parameters": nparams,
            "reference_sha256": sha256(a.reference.read_bytes()).hexdigest(),
            "archives_sha256": {k: sha256(v.read_bytes()).hexdigest() for k, v in files.items()}, "arms": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    for key in ("reference", "dallas", "unemployment", "industry", "later-unemployment", "later-industry", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
