#!/usr/bin/env python3
"""Benchmark stable packed sorting for authoritative integer clock offsets."""

import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import torch

from benchmark_kam_guarded_fp32_sdpa import variants
from benchmark_kam_radial_sdpa import labels
from benchmark_kam_typed_guard_cropping import PANELS
from kam_address_local_clock import fields, local_rank, rank_from_order
from run_kam_bitemporal_decoder import encode, fingerprint, make_decoder, record_order
from run_kam_bitemporal_noise import generate
from run_kam_clock_release_training import source_panels
from run_kam_packed_clock_inference import EAGER, evaluate, modify_unrelated, profile
from run_kam_temporal_clock_encoding import recode
from run_kam_typed_clock_guard import ENCODING, admit_order, guarded_tokens

B = 22
OFFSET = 1 << (B-1)
MASK = (1 << B)-1


def packed_rank(tokens, ticks, *, alias=False):
    """The caller supplies exact ordered int64 clock ticks within the signed 22-bit range."""
    address, _, available = fields(tokens, alias)
    assert ticks.shape == (*address.shape, 2) and ticks.dtype == torch.int64
    newest = MASK - (ticks + OFFSET)
    first = ((address << 1) | (~available).long()) << B
    key = ((first | newest[..., 0]) << B) | newest[..., 1]
    order = key.argsort(dim=1, stable=True)
    ordinal = torch.arange(tokens.shape[1], device=tokens.device)[None].expand_as(address)
    return rank_from_order(address, ticks, order, ordinal), available


def preprocess(raw, ticks):
    ranks, available = packed_rank(raw, ticks)
    values = recode(raw, ENCODING)
    values[..., 39] = torch.where(available, (ranks.float()+1).reciprocal(), values[..., 39])
    order, query = admit_order(raw)
    return values.gather(1, order[..., None].expand_as(values)), query


def integer_offsets(scene, permutation):
    q, aliases, facts, *_ = scene
    relative = torch.cat((aliases[..., 3:5], facts[..., 3:5]), dim=1) - q[:, None, 2:4]
    ranked = relative.gather(1, permutation[..., None].expand_as(relative))
    return torch.cat((ranked, torch.zeros_like(ranked[:, :1])), dim=1)


def audit(raw, ticks):
    assert ticks.abs().max().item() < OFFSET
    assert torch.equal(raw[..., -2:].sign(), ticks.sign().to(raw.dtype))
    for alias in (False, True):
        r, observed = local_rank(raw, alias=alias)
        s, admitted = packed_rank(raw, ticks, alias=alias)
        assert torch.equal(observed, admitted) and torch.equal(r[observed], s[observed])
    a = guarded_tokens(raw, ranked=True)
    b = preprocess(raw, ticks)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@torch.inference_mode()
def run(a):
    torch.set_float32_matmul_precision("highest")
    state = torch.load(a.weights, map_location="cpu", weights_only=True)
    assert state["seed"] == a.seed
    paths, archive = source_panels(a)
    archives = {}
    # The archives store whole source-date offsets with separate fixed units per clock.
    for name, records in archive.items():
        n = 0
        step = torch.tensor((24., 24. if name == "dallas" else 720.), device="cuda")
        for panel in records:
            for block in panel["data"][0].split(32):
                ticks = (block[..., -2:]*step).round().long()
                audit(block, ticks)
                n += len(block)
        archives[name] = n
    operators = {}
    for g, weights in state["models"].items():
        reference = make_decoder(f"{g}_without_position", a.seed, "cuda").eval()
        reference.load_state_dict(weights)
        operators[g] = variants(g, reference)["fp32_efficient"]
    arms = {}
    for d, n in PANELS:
        scene = generate(a.seed+d+127830, 64, d, "mixed", "cpu")
        varied = modify_unrelated(scene, a.seed+110639+d)
        order = record_order(len(scene[0]), d, a.seed+d+121575, "cpu")
        for name, panel in (("standard", scene), ("future_distractors", varied)):
            raw, *_ = encode(panel, order, hide_query_clock=True)
            ticks = integer_offsets(panel, order)
            assert torch.equal(raw[..., -2:], ticks.float()/24)
            assert torch.equal((raw[..., -2:]*24).round().long(), ticks)
            raw, ticks = raw.cuda(), ticks.cuda()
            audit(raw, ticks)
            reference = partial(guarded_tokens, ranked=True)
            start = (a.seed % (len(raw)//n))*n
            batch, metadata = raw[start:start+n], ticks[start:start+n]
            def prepare(x):
                return preprocess(x, metadata)
            call = {"rank_reference": lambda: local_rank(batch), "rank_packed": lambda: packed_rank(batch, metadata),
                    "prepare_reference": lambda: reference(batch), "prepare_packed": lambda: prepare(batch),
                    "prepare_reconstruct": lambda: preprocess(batch, (batch[..., -2:]*24).round().long())}
            matches = {}
            for g, (model, reader) in operators.items():
                method = "batch_max" if torch.cuda.get_device_name() != "NVIDIA B200" or g == "dot" else "full"
                def execute(x, pre, *, m=model, r=reader, layout=method):
                    return evaluate(m, EAGER, x, layout, preprocess=pre, readout=r)
                a_rows, b_rows = [], []
                for x, stamp in zip(raw.split(n), ticks.split(n), strict=True):
                    a_rows.append(execute(x, reference))
                    b_rows.append(execute(x, partial(preprocess, ticks=stamp)))
                before, after = torch.cat(a_rows), torch.cat(b_rows)
                assert torch.equal(before, after)
                matches[g] = {"layout": method, "tested": len(after),
                              "exact_readout_choices": int(labels(before).eq(labels(after)).all(-1).sum())}
                call[g+"/reference"] = partial(execute, batch, reference)
                call[g+"/packed"] = partial(execute, batch, prepare)
            timing = {k: profile(call[k], a.repeats) for k in Random(a.seed+519*d).sample(list(call), len(call))}
            arms[f"{d}/{name}"] = {"fixture_sha256": fingerprint(panel, order), "batch": n,
                                   "input_metadata_bytes": metadata.nbytes, "times": timing, "parity": matches}
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "weights_sha256": sha256(a.weights.read_bytes()).hexdigest(),
            "archive_inputs_sha256": {k: sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
            "input_parity_archive_counts": archives, "panels": arms}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("dallas", "unemployment", "industry", "later-unemployment", "later-industry", "weights", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--repeats", type=int, default=28)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
