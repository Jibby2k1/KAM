#!/usr/bin/env python3
"""Test learned event pointers on ordered durations and checked counterfactuals."""

import argparse
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from audit_kam_unseentimeqa import DAY, clocks, question_key
from audit_kam_unseentimeqa_serial import ANCHOR_END, TARGET, serial_events
from kam.model import KAMSequenceModel

MODES = ("local", "dot", "radial")
SCHEDULE = ((16, 64), (32, 32))
SCENARIOS = ("original", "sham", "intervention")


class ContextPointer(nn.Module):
    def __init__(self, score):
        super().__init__()
        self.encoder = KAMSequenceModel(task="regression", input_dim=6, d_model=64, num_heads=4, num_layers=2,
                                        num_supports=24, position_mode="sinusoidal", max_seq_len=129,
                                        score_type=score, use_memory=False, ffn_expansion=2,
                                        expose_memory_weights=False)
        self.encoder.readout = nn.Identity()
        self.key = nn.Linear(64, 64, bias=False)
        self.query = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        features, _ = self.encoder.encode(x)
        key = self.key(features[:, :-1])
        query = self.query(features[:, -1])
        return (key * query[:, None]).sum(-1) / 8


class LocalPointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(6, 192), nn.GELU(), nn.Linear(192, 192), nn.GELU(),
                                   nn.Linear(192, 192), nn.GELU(), nn.Linear(192, 1))

    def forward(self, x):
        return self.score(x[:, :-1]).squeeze(-1)


def encode(duration, anchor, offset):
    n = duration.shape[1]
    positions = torch.arange(n, device=duration.device)[None, :]
    relative = positions - anchor[:, None]
    spread = offset[:, None].expand_as(duration)
    local = torch.stack((duration / 64, relative / 32, spread / 1000, relative == 0,
                         torch.zeros_like(duration), torch.full_like(duration, n / 32, dtype=torch.float32)), -1)
    query = torch.stack((torch.zeros_like(offset), torch.zeros_like(offset), offset / 1000, torch.zeros_like(offset),
                         torch.ones_like(offset), torch.full_like(offset, n / 32, dtype=torch.float32)), -1)
    return torch.cat((local, query[:, None]), 1)


def clock(duration, anchor, offset):
    ends = duration.cumsum(-1)
    starts = ends - duration
    absolute = starts.gather(1, anchor[:, None]) + offset[:, None]
    return (ends <= absolute).sum(-1)


def sample(seed, n, count, maximum):
    rng = torch.Generator().manual_seed(seed)
    durations = torch.randint(1, maximum + 1, (count, n), generator=rng)
    anchor = torch.randint(0, n - 2, (count,), generator=rng)
    later = torch.rand(count, generator=rng)
    gold = anchor + 1 + (later * (n - 1 - anchor)).long()
    offsets = (torch.rand(count, generator=rng) * durations.gather(1, gold[:, None]).squeeze(1)).long()
    starts = durations.cumsum(-1) - durations
    offset = (starts.gather(1, gold[:, None]) - starts.gather(1, anchor[:, None])).squeeze(1) + offsets
    assert torch.equal(clock(durations, anchor, offset), gold)
    return durations, anchor, offset, gold


def sha(data):
    h = sha256()
    for tensor in data:
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def score(model, inputs, anchor):
    unallowed = torch.arange(inputs.shape[1] - 1, device=inputs.device)[None] < anchor[:, None]
    return model(inputs).masked_fill(unallowed, -torch.inf)


def fit(model, banks, steps, *, encoder=encode):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, fused=True)
    model.train()
    torch.cuda.synchronize()
    start = perf_counter()
    tokens, losses = 0, []
    for step in range(1, steps + 1):
        data, choices = banks[(step - 1) % len(banks)]
        index = choices[(step - 1) // len(banks)]
        duration, anchor, offset, gold = (a[index] for a in data)
        inputs = encoder(duration, anchor, offset)
        optimizer.param_groups[0]["lr"] = 0.0015 * min(step / 80, 1.0) * (
            0.2 + 0.8 * (1 + math.cos(math.pi * step / steps)) / 2)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(score(model, inputs, anchor), gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += inputs.shape[0] * inputs.shape[1]
        if step in (100, 400, 800, 1600, steps):
            assert torch.isfinite(loss)
            losses.append({"step": step, "loss": float(loss.detach())})
    torch.cuda.synchronize()
    return {"tokens": tokens, "seconds": perf_counter() - start, "losses": losses}


@torch.inference_mode()
def predict(models, data, *, encoder=encode):
    duration, anchor, offset, gold = data
    assert torch.equal(clock(duration, anchor, offset), gold)
    means = (anchor + torch.floor(offset / duration.float().mean(-1))).long().clamp(max=duration.shape[1] - 1)
    outputs = {"mean_duration": means.tolist(), "exact_prefix": gold.tolist()}
    for name, model in models.items():
        model.eval()
        blocks = zip(duration.split(32), anchor.split(32), offset.split(32), strict=True)
        chosen = torch.cat([score(model, encoder(d, a, o), a).argmax(-1) for d, a, o in blocks])
        outputs[name] = chosen.tolist()
    return outputs


def prepared_fixture(rows):
    groups = {}
    for row in rows:
        bank = row["variants"]["original"]["events"]
        original = serial_events(bank)
        anchor = row["anchor"]
        header, question = TARGET.split(row["question"], maxsplit=1)
        _, anchor_clause = ANCHOR_END.split(header, maxsplit=1)
        [anchor_clock] = clocks(anchor_clause)
        query = question_key(question)[1]
        elapsed = (query - anchor_clock + original[anchor].start) % DAY - original[anchor].start
        for key in SCENARIOS:
            scenario = row["variants"][key]
            durations = [event.end - event.start for event in serial_events(scenario["events"])]
            group = groups.setdefault(len(bank), [])
            group.append((row["id"], key, durations, anchor, elapsed, scenario["gold"]))
    return groups


def build_model(mode, seed):
    torch.manual_seed(seed)
    return (LocalPointer() if mode == "local" else ContextPointer(mode)).to("cuda")


def run(args):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    raw = args.fixture.read_bytes()
    rows = json.loads(raw)["rows"]
    assert len(rows) == 1173 and args.steps >= 100 and args.steps % 2 == 0
    banks, fingerprints = [], []
    for i, (depth, batch) in enumerate(SCHEDULE):
        data = sample(args.seed + 22603 + i, depth, args.bank, 90)
        rng = torch.Generator().manual_seed(args.seed + 31607 + i)
        choices = torch.randint(args.bank, (args.steps // 2, batch), generator=rng)
        fingerprints.append(sha((*data, choices)))
        banks.append((tuple(x.to("cuda") for x in data), choices.to("cuda")))
    models = {mode: build_model(mode, args.seed) for mode in MODES}
    training = {}
    for mode in (*MODES[args.order:], *MODES[:args.order]):
        model = models[mode]
        training[mode] = {"parameters": sum(p.numel() for p in model.parameters()), **fit(model, banks, args.steps)}
        print(mode, training[mode]["seconds"], flush=True)

    conditions = ((16, 90), (32, 90), (64, 90), (128, 90), (64, 45), (128, 22))
    synthetic = {}
    for i, (depth, maximum) in enumerate(conditions):
        data = sample(args.seed + 612991 + i, depth, 512, maximum)
        synthetic[f"{depth}/{maximum}"] = {"sha256": sha(data), "gold": data[-1].tolist(),
                                           "predictions": predict(models, tuple(x.to("cuda") for x in data))}
    records = {}
    for events, groups in prepared_fixture(rows).items():
        keys = [(r[0], r[1]) for r in groups]
        data = tuple(torch.tensor(col, device="cuda") for col in zip(*(r[2:] for r in groups), strict=True))
        scores = predict(models, data)
        for i, (row, condition) in enumerate(keys):
            records.setdefault(row, {})[condition] = {k: ranks[i] for k, ranks in scores.items()}
    return {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "seed": args.seed,
            "steps": args.steps, "bank": args.bank, "bank_sha256": fingerprints,
            "fixture_sha256": sha256(raw).hexdigest(), "training": training, "synthetic": synthetic,
            "archived": [{"id": row["id"], "scenarios": records[row["id"]]} for row in rows]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--order", default=0, choices=(0, 1, 2), type=int)
    p.add_argument("--bank", default=16_384, type=int)
    p.add_argument("--steps", default=2400, type=int)
    args = p.parse_args()
    args.output.write_bytes(gzip.compress(json.dumps(run(args), separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
