#!/usr/bin/env python3
"""Test scale-aware coordinates and equal-token length exposure for event pointers."""

import argparse
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MethodType

import torch

from run_kam_duration_pointer import ContextPointer, SCHEDULE, SCENARIOS, encode, fit, predict, prepared_fixture, sample, sha

WIDE = ((16, 70), (32, 35), (64, 16), (128, 7))
CONTROL = {
    "baseline": (False, False, SCHEDULE),
    "scaled_features": (True, False, SCHEDULE),
    "scaled_positions": (False, True, SCHEDULE),
    "scaled_both": (True, True, SCHEDULE),
    "wide_baseline": (False, False, WIDE),
    "wide_scaled_both": (True, True, WIDE),
}


def scaled_features(duration, anchor, offset):
    n = duration.shape[1]
    relative = torch.arange(n, device=duration.device)[None, :] - anchor[:, None]
    durations = duration / duration.float().mean(-1, keepdim=True)
    query_fraction = offset / ((relative >= 0) * duration).sum(-1)
    repeated = query_fraction[:, None].expand_as(durations)
    events = torch.stack((durations, relative / (n - 1), repeated, relative == 0,
                          torch.zeros_like(duration), torch.ones_like(duration)), -1)
    zero, one = torch.zeros_like(offset), torch.ones_like(offset)
    query = torch.stack((zero, zero, query_fraction, zero, one, one), -1)
    return torch.cat((events, query[:, None]), 1)


def scaled_position_encoding(encoder, length, device, dtype):
    positions = torch.linspace(0, 32, length, device=device, dtype=dtype)
    exponent = (-math.log(10000.0) / encoder.d_model) * torch.arange(0, encoder.d_model, 2, device=device, dtype=dtype)
    angles = positions[:, None] * torch.exp(exponent)
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[None]


def scaled_relative_bias(bias, length, device):
    positions = torch.linspace(0, 32, length, device=device)
    lags = (positions[:, None] - positions[None, :]).clamp(0, bias.max_lag - 1)
    lo = lags.long()
    hi = (lo + 1).clamp_max(bias.max_lag - 1)
    return torch.lerp(bias.bias[:, lo], bias.bias[:, hi], lags - lo)


def pointer(seed, positions):
    torch.manual_seed(seed)
    net = ContextPointer("dot")
    if positions:
        net.encoder._position_encoding = MethodType(scaled_position_encoding, net.encoder)
        for block in net.encoder.blocks:
            bias = block.context.relative_bias
            bias.forward = MethodType(scaled_relative_bias, bias)
    return net.to("cuda")


def banks(seed, n, steps, schedule):
    out, stamps = [], []
    assert steps % len(schedule) == 0
    for depth, batch in schedule:
        data = sample(seed + 22603 + depth, depth, n, 90)
        rng = torch.Generator().manual_seed(seed + 31607 + depth)
        controls = torch.randint(n, (steps // len(schedule), batch), generator=rng)
        stamps.append(sha((*data, controls)))
        out.append((tuple(x.to("cuda") for x in data), controls.to("cuda")))
    return out, stamps


def run(args):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    payload = args.fixture.read_bytes()
    rows = json.loads(payload)["rows"]
    assert len(rows) == 1173 and args.steps >= 100 and args.steps % 4 == 0
    series = {k: banks(args.seed, args.bank, args.steps, k) for k in (SCHEDULE, WIDE)}
    shapes = ((16, 90), (32, 90), (64, 90), (128, 90), (64, 45), (128, 22), (256, 90))
    synthetic = {}
    for i, (depth, maximum) in enumerate(shapes):
        data = sample(args.seed + 612991 + i, depth, 512, maximum)
        synthetic[f"{depth}/{maximum}"] = {"sha256": sha(data), "gold": data[-1].tolist(),
                                           "data": tuple(x.to("cuda") for x in data), "predictions": {}}
    archive = {}
    for events, entries in prepared_fixture(rows).items():
        keys = [(r[0], r[1]) for r in entries]
        data = tuple(torch.tensor(col, device="cuda") for col in zip(*(r[2:] for r in entries), strict=True))
        archive[events] = (keys, data)
    modes = tuple(CONTROL)
    outputs, training = {}, {}
    for name in (*modes[args.order:], *modes[:args.order]):
        features, positions, schedule = CONTROL[name]
        transform = scaled_features if features else encode
        model = pointer(args.seed, positions)
        numeric, hashes = series[schedule]
        training[name] = {"schedule": schedule, "bank_sha256": hashes,
                          **fit(model, numeric, args.steps, encoder=transform)}
        print(name, training[name]["seconds"], flush=True)
        for panel in synthetic.values():
            scores = predict({name: model}, panel["data"], encoder=transform)
            panel["predictions"].update(scores)
        for keys, data in archive.values():
            scores = predict({name: model}, data, encoder=transform)
            for i, (row, condition) in enumerate(keys):
                outputs.setdefault(row, {}).setdefault(condition, {}).update({k: v[i] for k, v in scores.items()})
    for panel in synthetic.values():
        del panel["data"]
    return {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "seed": args.seed,
            "steps": args.steps, "bank": args.bank, "fixture_sha256": sha256(payload).hexdigest(),
            "training": training, "synthetic": synthetic,
            "archived": [{"id": row["id"], "scenarios": {s: outputs[row["id"]][s] for s in SCENARIOS}}
                         for row in rows]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--order", default=0, type=int, choices=range(len(CONTROL)))
    p.add_argument("--steps", default=2400, type=int)
    p.add_argument("--bank", default=16_384, type=int)
    args = p.parse_args()
    args.output.write_bytes(gzip.compress(json.dumps(run(args), separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
