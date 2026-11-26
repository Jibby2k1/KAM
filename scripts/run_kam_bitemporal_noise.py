#!/usr/bin/env python3
"""Compare learned one-pass and sequential pointers under controlled revision noise."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn.functional import cross_entropy

PROFILES = ("clean", "irrelevant", "homonym", "future_effective", "late_arrival", "stale", "mixed")
CORRECTIONS = ("older_correction", "late_correction", "correction_mix")
ARMS = ("exact", "known_only", "effective_only", "sequential", "one_pass_pair", "sequential_with_gold_alias")


def generate(seed, examples, depth, profile, device):
    """Create matched histories with additive, label-preserving distractors."""
    rng = torch.Generator().manual_seed(seed)

    def samples(low, high, columns=1):
        return torch.randint(low, high, (examples, columns), generator=rng)

    if depth < 12 or (profile == "clean" and depth != 12):
        raise ValueError("all histories start with 12 records per bank")
    workspace, alias, entity = samples(0, 8), samples(0, 32), samples(0, 64)
    value = samples(0, 128)
    effective = samples(38, 96)
    known = effective + samples(0, 17)
    query = torch.cat((workspace, alias, effective, known), dim=1)
    base_alias_e, base_fact_e = effective - samples(2, 16), effective - samples(2, 16)
    base_alias_k, base_fact_k = known - samples(1, 16), known - samples(1, 16)
    truth_alias = torch.stack((workspace, alias, entity, base_alias_e, base_alias_k), dim=-1)
    truth_fact = torch.stack((workspace, entity, value, base_fact_e, base_fact_k), dim=-1)

    def block(kind, n):
        w = workspace.expand(-1, n)
        a = alias.expand(-1, n)
        e = entity.expand(-1, n)
        other_w = (w + samples(1, 8, n)) % 8
        other_a = (a + samples(1, 32, n)) % 32
        other_e = (e + samples(1, 64, n)) % 64
        false_value = (value + samples(1, 128, n)) % 128
        if kind == "irrelevant":
            salias = (other_w, other_a, other_e, effective + samples(-24, 24, n), known + samples(-24, 24, n))
            sfact = (other_w, other_e, false_value, effective + samples(-24, 24, n), known + samples(-24, 24, n))
        elif kind == "homonym":
            salias = (other_w, a, other_e, effective - samples(0, 2, n), known - samples(0, 2, n))
            sfact = (other_w, e, false_value, effective - samples(0, 2, n), known - samples(0, 2, n))
        elif kind == "future_effective":
            salias = (w, a, other_e, effective + samples(1, 17, n), known - samples(0, 10, n))
            sfact = (w, e, false_value, effective + samples(1, 17, n), known - samples(0, 10, n))
        elif kind == "late_arrival":
            salias = (w, a, other_e, effective - samples(0, 2, n), known + samples(1, 17, n))
            sfact = (w, e, false_value, effective - samples(0, 2, n), known + samples(1, 17, n))
        elif kind == "stale":
            salias = (w, a, other_e, base_alias_e - samples(1, 17, n), known - samples(0, 8, n))
            sfact = (w, e, false_value, base_fact_e - samples(1, 17, n), known - samples(0, 8, n))
        elif kind == "older_correction":
            salias = (w, a, other_e, base_alias_e.expand(-1, n), base_alias_k - samples(1, 17, n))
            sfact = (w, e, false_value, base_fact_e.expand(-1, n), base_fact_k - samples(1, 17, n))
        elif kind == "late_correction":
            salias = (w, a, other_e, base_alias_e.expand(-1, n), known + samples(1, 17, n))
            sfact = (w, e, false_value, base_fact_e.expand(-1, n), known + samples(1, 17, n))
        else:
            raise ValueError(kind)
        return torch.stack(salias, -1), torch.stack(sfact, -1)

    blocks = [(truth_alias, truth_fact), block("irrelevant", 8), block("stale", 3)]
    extra = depth - 12
    if extra:
        kinds = PROFILES[1:-1] if profile == "mixed" else (profile,)
        if profile == "correction_mix":
            kinds = (*PROFILES[1:-1], *CORRECTIONS[:2])
        for i, kind in enumerate(kinds):
            count = (i + 1) * extra // len(kinds) - i * extra // len(kinds)
            if count:
                blocks.append(block(kind, count))
    banks = [torch.cat(parts, dim=1) for parts in zip(*blocks, strict=True)]
    shuffled, gold = [], []
    for bank in banks:
        order = torch.rand((examples, depth), generator=rng).argsort(-1)
        gold.append(order.eq(0).long().argmax(-1))
        shuffled.append(bank.gather(1, order[..., None].expand(-1, -1, bank.shape[-1])))
    return tuple(t.to(device) for t in (query, *shuffled, *gold, value[:, 0]))


def clock_filter(bank, q, timing):
    if timing == "known_only":
        return bank[..., 4] <= q[:, None, 3]
    if timing == "effective_only":
        return bank[..., 3] <= q[:, None, 2]
    return (bank[..., 3] <= q[:, None, 2]) & (bank[..., 4] <= q[:, None, 3])


def exact(query, aliases, facts, clock):
    workspace, name = query[:, None, 0], query[:, None, 1]
    alias_valid = (aliases[..., 0] == workspace) & (aliases[..., 1] == name) & clock_filter(aliases, query, clock)
    timeline = aliases[..., 3] * 256 + aliases[..., 4]
    alias = timeline.masked_fill(~alias_valid, -1).argmax(-1)
    entity = aliases[..., 2].gather(1, alias[:, None])
    fact_valid = (facts[..., 0] == workspace) & (facts[..., 1] == entity) & clock_filter(facts, query, clock)
    timeline = facts[..., 3] * 256 + facts[..., 4]
    ranked_fact = timeline.masked_fill(~fact_valid, -1).argmax(-1)
    return (alias.masked_fill(~alias_valid.any(-1), -1),
            ranked_fact.masked_fill(~(alias_valid.any(-1) & fact_valid.any(-1)), -1))


def scalar_oracle(query, aliases, facts):
    """Independent oracle for a small shadow run, using tuple ordering."""
    a_out, f_out = [], []
    for q, a_bank, f_bank in zip(query.tolist(), aliases.tolist(), facts.tolist(), strict=True):
        w, alias, date, cutoff = q
        aidx = max((i for i, row in enumerate(a_bank)
                    if row[0] == w and row[1] == alias and row[3] <= date and row[4] <= cutoff),
                   key=lambda i: tuple(a_bank[i][3:]))
        entity = a_bank[aidx][2]
        fidx = max((i for i, row in enumerate(f_bank)
                    if row[0] == w and row[1] == entity and row[3] <= date and row[4] <= cutoff),
                   key=lambda i: tuple(f_bank[i][3:]))
        a_out.append(aidx)
        f_out.append(fidx)
    return a_out, f_out


def features(query, aliases, facts, *, include_join=True):
    a = torch.stack((aliases[..., 0] == query[:, None, 0], aliases[..., 1] == query[:, None, 1],
                     (aliases[..., 3] - query[:, None, 2]) / 24,
                     (aliases[..., 4] - query[:, None, 3]) / 24), -1).float()
    f = torch.stack((facts[..., 0] == query[:, None, 0],
                     (facts[..., 3] - query[:, None, 2]) / 24,
                     (facts[..., 4] - query[:, None, 3]) / 24), -1).float()
    join = aliases[..., 2, None] == facts[:, None, :, 1] if include_join else None
    return a, f, join


class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alias = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))
        self.fact = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))
        self.join_bias = nn.Parameter(torch.tensor(4.0))

    def local(self, af, ff):
        return self.alias(af).squeeze(-1), self.fact(ff).squeeze(-1)

    def pair(self, a, f, join):
        return a[..., None] + f[:, None, :] + self.join_bias * join

    def condition(self, f, join, selected):
        row = torch.arange(len(f), device=f.device)
        return f + self.join_bias * join[row, selected]


def fit(a, mode, device):
    torch.manual_seed(a.seed)
    model = Pointer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, fused=device.type == "cuda")
    rng = torch.Generator(device=device).manual_seed(a.seed + 947)
    train = []
    # The same frozen, mixed-depth fitting batches serve both pointer arms.
    mixed = "correction_mix" if getattr(a, "train_regime", "baseline") == "corrections" else "mixed"
    for i, profile in enumerate(("clean", mixed)):
        scene = generate(a.seed + 2000 + i, a.train_examples // 2, 12 if i == 0 else 20, profile, device)
        q, aliases, facts, good_a, good_f, _ = scene
        train.append((*features(q, aliases, facts), good_a, good_f))
    for _ in range(a.epochs):
        for af, ff, join, good_a, good_f in train:
            for batch in torch.randperm(len(good_a), device=device, generator=rng).split(a.batch):
                av, fv = model.local(af[batch], ff[batch])
                if mode == "one_pass_pair":
                    logits = model.pair(av, fv, join[batch]).flatten(start_dim=1)
                    label = good_a[batch] * fv.shape[1] + good_f[batch]
                    loss = cross_entropy(logits, label)
                else:
                    fact_logits = model.condition(fv, join[batch], good_a[batch])
                    loss = cross_entropy(av, good_a[batch]) + cross_entropy(fact_logits, good_f[batch])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            assert torch.isfinite(loss)
    return model.eval(), float(loss.detach())


def bounded_pairs(model, a, f, join, chunk):
    if chunk >= a.shape[1]:
        return model.pair(a, f, join).flatten(start_dim=1).argmax(-1)
    best = torch.full((len(a),), -torch.inf, device=a.device)
    index = torch.zeros((len(a),), device=a.device, dtype=torch.long)
    for start in range(0, a.shape[1], chunk):
        sl = slice(start, start + chunk)
        values, local = model.pair(a[:, sl], f, join[:, sl]).flatten(start_dim=1).max(-1)
        update = values > best
        best = torch.maximum(values, best)
        index = torch.where(update, local + start * f.shape[1], index)
    return index


@torch.inference_mode()
def measure(scene, models, batch):
    q, aliases, facts, gold_a, gold_f, payload = scene
    values = {}
    for arm, clock in (("exact", "both"), ("known_only", "known_only"), ("effective_only", "effective_only")):
        values[arm] = exact(q, aliases, facts, clock)
    candidates = {name: [[], []] for name in ARMS[3:]}
    for start in range(0, len(q), batch):
        sl = slice(start, start + batch)
        af, ff, join = features(q[sl], aliases[sl], facts[sl])
        seq = models["sequential"]
        av, fv = seq.local(af, ff)
        alias = av.argmax(-1)
        fact = seq.condition(fv, join, alias).argmax(-1)
        teacher = seq.condition(fv, join, gold_a[sl]).argmax(-1)
        pair = models["one_pass_pair"]
        ap, fp = pair.local(af, ff)
        winner = bounded_pairs(pair, ap, fp, join, 32)
        if start == 0:
            direct = bounded_pairs(pair, ap, fp, join, ap.shape[1])
            assert torch.equal(winner, direct)
        for arm, (left, right) in zip(ARMS[3:],
                                      ((alias, fact), (winner // fp.shape[1], winner % fp.shape[1]), (gold_a[sl], teacher)),
                                      strict=True):
            candidates[arm][0].append(left)
            candidates[arm][1].append(right)
    for name, parts in candidates.items():
        values[name] = (torch.cat(parts[0]), torch.cat(parts[1]))
    assert torch.equal(values["exact"][0], gold_a) and torch.equal(values["exact"][1], gold_f)
    descriptions = {}
    for name, (left, right) in values.items():
        av = aliases[torch.arange(len(q), device=q.device), left.clamp_min(0)]
        fv = facts[torch.arange(len(q), device=q.device), right.clamp_min(0)]
        has_alias, has_fact = left >= 0, right >= 0
        correct_a, correct_f = left == gold_a, right == gold_f
        descriptions[name] = {
            "alias": int(correct_a.sum()), "fact": int(correct_f.sum()), "joint": int((correct_a & correct_f).sum()),
            "payload": int((has_fact & fv[:, 2].eq(payload)).sum()),
            "same_entity": int((has_alias & has_fact & av[:, 2].eq(fv[:, 1])).sum()),
            "abstains": int((~has_fact).sum()),
            "future_effective": int(((has_alias & (av[:, 3] > q[:, 2])) | (has_fact & (fv[:, 3] > q[:, 2]))).sum()),
            "late_arrival": int(((has_alias & (av[:, 4] > q[:, 3])) | (has_fact & (fv[:, 4] > q[:, 3]))).sum()),
            "wrong_workspace": int(((has_alias & (av[:, 0] != q[:, 0])) | (has_fact & (fv[:, 0] != q[:, 0]))).sum()),
        }
    return descriptions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--train-examples", type=int, default=8192)
    p.add_argument("--test-examples", type=int, default=2048)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--train-regime", choices=("baseline", "corrections"), default="baseline")
    p.add_argument("--correction-panel", action="store_true")
    a = p.parse_args()
    assert a.train_examples >= 2 and a.train_examples % 2 == 0 and a.test_examples >= 16
    torch.set_float32_matmul_precision("highest")
    device = torch.device(a.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    start = perf_counter()
    models = {arm: fit(a, arm, device) for arm in ("sequential", "one_pass_pair")}
    train_done = perf_counter()
    rows = []
    schedule = [(12, "clean"), *((32, profile) for profile in PROFILES[1:]), (64, "mixed"), (128, "mixed")]
    if a.correction_panel:
        schedule.extend((depth, profile) for depth in (32, 64, 128) for profile in CORRECTIONS)
    for depth, profile in schedule:
        scene = generate(a.seed + 91_237, a.test_examples, depth, profile, device)
        q, aliases, facts, gold_a, gold_f, target = scene
        independent = scalar_oracle(*(t[:16].cpu() for t in (q, aliases, facts)))
        assert independent == (gold_a[:16].tolist(), gold_f[:16].tolist())
        arms = measure(scene, {arm: model for arm, (model, _) in models.items()}, a.batch)
        raw = b"".join(t.cpu().numpy().tobytes() for t in (q, gold_a, gold_f, target))
        rows.append({"profile": profile, "depth_per_bank": depth, "matched_sha256": sha256(raw).hexdigest(),
                     "query_payload_sha256": sha256(q.cpu().numpy().tobytes() + target.cpu().numpy().tobytes()).hexdigest(),
                     "count": a.test_examples, "arms": arms})
    assert len({r["query_payload_sha256"] for r in rows}) == 1
    payload = {"seed": a.seed, "epochs": a.epochs, "train_examples": a.train_examples,
               "train_regime": a.train_regime,
               "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
               "torch": str(torch.__version__), "rows": rows,
               "fitting": {arm: {"loss": loss, "parameters": sum(p.numel() for p in model.parameters())}
                           for arm, (model, loss) in models.items()},
               "fit_seconds": train_done - start, "run_seconds": perf_counter() - start}
    a.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
