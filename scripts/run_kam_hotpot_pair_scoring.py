#!/usr/bin/env python3
"""Learn a symmetric score for every candidate document pair in one shot."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from audit_kam_natural_data import digest, words
from run_kam_hotpot_head_to_head import adapter, encode, reader, route
from run_kam_natural_feasibility import Encoder
from run_kam_natural_router import score, to_device


def links(cases):
    """Precompute exact title mentions in the first 120 word tokens."""
    rows = []
    for case in cases:
        titles = [" " + " ".join(words(title)) + " " for title, _ in case["context"]]
        bodies = [" " + " ".join(words(" ".join(sentences))[:120]) + " " for _, sentences in case["context"]]
        rows.append([[len(title.strip()) >= 3 and title in body for title in titles] for body in bodies])
    return torch.tensor(rows, dtype=torch.bool)


def features(query, keys, mentions, index):
    a, b = index
    scores = score(query, keys)
    first, second = scores[:, a], scores[:, b]
    forward, back = mentions[:, a, b], mentions[:, b, a]
    x = torch.stack([
        (first + second), (first - second).abs(),
        (keys[:, a] * keys[:, b]).sum(-1),
        (forward | back).float(), (forward & back).float(),
        (forward * (first - second) + back * (second - first)),
    ], dim=-1)
    return (first + second) * 20, x


class PairScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(6, 24), nn.SiLU(), nn.Linear(24, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, base, pair_features):
        return base + self.layers(pair_features).squeeze(-1)


def training(args, spec, model, index, train_mentions):
    cached = torch.load(args.training_cache, map_location="cpu", weights_only=True)
    assert cached["partition_sha256"] == json.loads(args.partition.read_text())["training_partition_sha256"]
    q, k, gold = to_device(cached["training"])
    with torch.no_grad():
        base, x = features(model(q), k, train_mentions, index)
        positives = gold[:, index[0]] & gold[:, index[1]]
        assert positives.sum(dim=-1).eq(1).all()
        labels = positives.float().argmax(dim=-1)
    size = spec["batch"]
    seed = spec["seeds"][args.shard]
    torch.manual_seed(seed + 100_000)
    pair = PairScorer().cuda()
    optimizer = torch.optim.AdamW(pair.parameters(), lr=spec["pair_learning_rate"], fused=True)
    rng = torch.Generator(device="cuda").manual_seed(seed + 140_000)
    torch.cuda.synchronize()
    started = perf_counter()
    for _ in range(spec["pair_epochs"]):
        for indices in torch.randperm(len(q), generator=rng, device="cuda").split(size):
            loss = cross_entropy(pair(base[indices], x[indices]), labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    return pair.eval(), perf_counter() - started


@torch.inference_mode()
def selected_pairs(query, keys, mentions, pair, index):
    base, x = features(query, keys, mentions, index)
    winner = pair(base, x).argmax(-1)
    a, b = index[:, winner]
    scores = score(query, keys)
    row = torch.arange(len(query), device=query.device)
    first = torch.where(scores[row, a] >= scores[row, b], a, b)
    second = torch.where(first == a, b, a)
    return torch.stack((first, second), dim=-1)


def evaluate(args, spec, cases):
    data = torch.load(args.fresh_cache, map_location="cpu", weights_only=True)
    link_cache = torch.load(args.link_cache, map_location="cpu", weights_only=True)
    assert data["partition_sha256"] == link_cache["partition_sha256"] == digest(args.partition)
    assert data["ids"] == [row["_id"] for row in cases]
    started = perf_counter()
    model = adapter(args, spec)
    index = torch.triu_indices(10, 10, 1, device="cuda")
    pair, training_seconds = training(args, spec, model, index, link_cache["training"].cuda())
    queries, keys = data["queries"].cuda(), data["keys"].cuda()
    mentions = link_cache["transfer"].cuda()
    with torch.inference_mode():
        adapted = model(queries)
        picks = selected_pairs(adapted, keys, mentions, pair, index)
        # Comparing unordered top pairs isolates permutation invariance
        # from the reader's evidence-presentation ordering.
        rev = torch.arange(9, -1, -1, device="cuda")
        reverse = rev[selected_pairs(adapted, keys[:, rev], mentions[:, rev][:, :, rev], pair, index)]
        agreement = int((picks.sort(-1).values == reverse.sort(-1).values).all(-1).sum())
        assert agreement == len(cases)
        torch.cuda.synchronize()
        begin = perf_counter()
        for _ in range(20):
            picks = selected_pairs(adapted, keys, mentions, pair, index)
        torch.cuda.synchronize()
        pair_seconds = (perf_counter() - begin) / 20
    encoder = Encoder(args.assets / "encoder", 64)
    selected, timing = route(encoder, model, cases, queries, keys, include_frozen=False)
    del encoder
    selected["learned_pair"] = picks.tolist()
    selected["gold"] = [
        [i for i, (title, _) in enumerate(case["context"])
         if title in {gold for gold, _ in case["supporting_facts"]}]
        for case in cases
    ]
    records, unique = reader(args, cases, selected)
    args.output.write_text(json.dumps({
        "status": "completed", "scope": "fresh public development", "seed": spec["seeds"][args.shard],
        "partition_sha256": digest(args.partition), "config_sha256": digest(args.config),
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "models": json.loads((args.assets / "revisions.json").read_text()),
        "training_seconds_pair": training_seconds, "pair_parameters": sum(p.numel() for p in pair.parameters()),
        "pair_permutation_agreement": agreement, "timing": {**timing, "learned_pair_rank_seconds": pair_seconds},
        "elapsed_seconds": perf_counter() - started, "unique_reader_prompts": unique,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "records": records,
    }, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("encode", "run"))
    for arg in ("source", "partition", "assets", "training-cache", "fresh-cache", "link-cache", "config"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--shard", type=int)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    manifest = json.loads(a.partition.read_text())
    spec = json.loads(a.config.read_text())
    assert manifest["source_sha256"] == digest(a.source)
    source = {row["_id"]: row for row in json.loads(a.source.read_text())}
    cases = [source[id_] for id_ in manifest["ids"]]
    if a.mode == "encode":
        encode(a, cases)
        train = torch.load(a.training_cache, map_location="cpu", weights_only=True)["training"]
        assert not set(train["ids"]) & set(manifest["ids"])
        torch.save({"training": links([source[id_] for id_ in train["ids"]]), "transfer": links(cases),
                    "partition_sha256": digest(a.partition)}, a.link_cache)
    else:
        evaluate(a, spec, cases)


if __name__ == "__main__":
    main()
