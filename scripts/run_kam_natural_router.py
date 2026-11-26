#!/usr/bin/env python3
"""Frozen-encoder, candidate-relative hard routing on title-disjoint development data."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn.functional import log_softmax, normalize

from audit_kam_natural_data import digest
from run_kam_natural_feasibility import Encoder, candidate_texts


class QueryAdapter(nn.Module):
    def __init__(self, width, bottleneck, nonlinear):
        super().__init__()
        self.input = nn.Linear(width, bottleneck, bias=False)
        self.output = nn.Linear(bottleneck, width, bias=False)
        nn.init.zeros_(self.output.weight)
        self.nonlinear = nonlinear

    def forward(self, query):
        hidden = self.input(query)
        return normalize(query + self.output(hidden.tanh() if self.nonlinear else hidden), dim=-1)


def encode(args):
    started = perf_counter()
    partition = json.loads(args.partition.read_text())
    assert digest(args.source) == partition["source_sha256"]
    frozen = json.loads((args.assets / "revisions.json").read_text())
    encoder = Encoder(args.assets / "encoder", args.batch)
    all_cases = {c["_id"]: c for c in json.loads(args.source.read_text())}
    cache = {"asset_revisions": frozen, "partition_sha256": digest(args.partition)}
    for split in ("training", "transfer"):
        cases = [all_cases[key] for key in partition[split]]
        queries = encoder([c["question"] for c in cases], "query").cpu()
        keys = encoder([s for case in cases for s in candidate_texts(case)], "passage").reshape(len(cases), 10, -1).cpu()
        masks = []
        for case in cases:
            support = {title for title, _ in case["supporting_facts"]}
            masks.append([title in support for title, _ in case["context"]])
        mask = torch.tensor(masks)
        assert (mask.sum(-1) == 2).all()
        cache[split] = {"ids": partition[split], "queries": queries, "keys": keys, "gold": mask}
    torch.save(cache, args.cache)
    print(json.dumps({"encoder_seconds": perf_counter() - started, "device": torch.cuda.get_device_name(),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "asset_revisions": frozen}))


def to_device(data, device="cuda"):
    return tuple(data[name].to(device) for name in ("queries", "keys", "gold"))


def score(query, keys):
    return torch.einsum("bd,bnd->bn", query, keys)


@torch.inference_mode()
def evaluate(query, keys, gold, ids):
    started = perf_counter()
    scores = score(query, keys)
    order = scores.argsort(dim=-1, descending=True, stable=True)
    order2 = order[:, :2]
    # A candidate-relative route must be invariant to moving the candidate slots.
    permutation = torch.arange(keys.shape[1] - 1, -1, -1, device=keys.device)
    reversed_top = permutation[score(query, keys[:, permutation]).argsort(dim=-1, descending=True)[:, :2]]
    ties = (scores.gather(1, order[:, 1:2]) == scores.gather(1, order[:, 2:3])).sum().item()
    invariant = (torch.sort(reversed_top, -1).values == torch.sort(order2, -1).values).all(-1).sum().item()
    if len(ids) - invariant > ties:
        raise ValueError("changed rank after candidate permutation without a reported tie")
    top2 = gold.gather(1, order2).cpu().tolist()
    selected = order2.cpu().tolist()
    return {"elapsed_seconds": perf_counter() - started, "top2_permutation_agreement": invariant,
            "top2_exact_boundary_ties": ties,
            "examples": [{"id": id_, "prediction": row, "first_is_support": hits[0], "both_support": all(hits)}
                         for id_, row, hits in zip(ids, selected, top2)]}


def fit(args):
    spec = json.loads(args.config.read_text())
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    partition = json.loads(args.partition.read_text())
    assert cache["partition_sha256"] == digest(args.partition)
    assert all(cache[key]["ids"] == partition[key] for key in ("training", "transfer"))
    train_q, train_k, train_gold = to_device(cache["training"])
    test_q, test_k, test_gold = to_device(cache["transfer"])
    seed = spec["seeds"][args.shard]
    ids = cache["transfer"]["ids"]
    arms = {"frozen": evaluate(test_q, test_k, test_gold, ids)}
    order = ("affine", "nonlinear") if args.shard % 2 == 0 else ("nonlinear", "affine")
    for name in order:
        torch.manual_seed(seed)
        model = QueryAdapter(train_q.shape[-1], spec["bottleneck"], name == "nonlinear").cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], fused=True)
        rng = torch.Generator(device="cuda").manual_seed(seed + 40_000)
        torch.cuda.synchronize()
        start = perf_counter()
        losses = []
        for _ in range(spec["epochs"]):
            for ids_ in torch.randperm(len(train_q), generator=rng, device="cuda").split(spec["batch"]):
                scores = score(model(train_q[ids_]), train_k[ids_]) * spec["score_scale"]
                loss = -(log_softmax(scores, dim=-1) * train_gold[ids_]).sum(-1).mean() / 2
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(loss.detach())
        torch.cuda.synchronize()
        training_seconds = perf_counter() - start
        with torch.inference_mode():
            arms[name] = {"parameters": sum(p.numel() for p in model.parameters()),
                          "training_seconds": training_seconds, "last_batch_losses": torch.stack(losses).tolist(),
                          **evaluate(model(test_q), test_k, test_gold, ids)}
    output = {"status": "completed", "scope": "development transfer; no answer-quality endpoint",
              "seed": seed, "config": spec, "config_sha256": digest(args.config),
              "partition_sha256": cache["partition_sha256"], "asset_revisions": cache["asset_revisions"],
              "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "arms": arms}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("encode", "fit"))
    p.add_argument("--partition", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--shard", type=int)
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    {"encode": encode, "fit": fit}[a.mode](a)


if __name__ == "__main__":
    main()
