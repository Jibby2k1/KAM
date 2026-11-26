#!/usr/bin/env python3
"""Refit matched pair heads on archived pools with prespecified feature groups."""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch.nn.functional import cross_entropy

from audit_kam_natural_data import digest
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_fullwiki_support_links import ContextScorer, relation_features
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import prepared
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder

AFFINITY = (0, 1, 2)
LINK = (3, 4, 5)
COVERAGE = (6, 7, 8)
MENTION = (9, 10)
GROUPS = {
    "refit_full": (*AFFINITY, *LINK, *COVERAGE, *MENTION),
    "refit_no_mention": (*AFFINITY, *LINK, *COVERAGE),
    "refit_no_coverage": (*AFFINITY, *LINK, *MENTION),
    "refit_without_links": (*AFFINITY, *COVERAGE),
    "refit_plain": (*AFFINITY, *LINK),
    "refit_affinity_only": AFFINITY,
}


@torch.no_grad()
def load_inputs(a, fixture, recorded, cfg):
    cases = {row["id"]: row for row in fixture["rows"]["training"]}
    assert set(cases) == {row["id"] for row in recorded["rows"]}
    torch.set_float32_matmul_precision("highest")
    index = FullWikiIndex(a.inputs, read(a.inputs / "manifest.json"), cfg)
    model, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    encoder = Encoder(a.assets / "encoder", 64)
    queries = encoder([cases[r["id"]]["question"] for r in recorded["rows"]], "query")
    depth = fixture["training_depth"]
    pairs = torch.triu_indices(depth, depth, 1)
    indices = pairs.cuda()
    base, features, labels = [], [], []
    pair_slots = {tuple(pair): i for i, pair in enumerate(pairs.T.tolist())}
    for i, row in enumerate(recorded["rows"]):
        q = cases[row["id"]]
        ranks = row["row_ids"]
        assert len(set(ranks)) == depth
        gold = tuple(slot for slot, did in enumerate(ranks) if did in q["gold_ids"])
        label = pair_slots[gold]
        example = index.candidates(q, ranks)
        keys = index.bank[torch.tensor(ranks, device="cuda")].unsqueeze(0)
        graph, context = relation_features(example, pairs)
        reference, inputs, _ = prepared(model(queries[i:i+1]), keys, graph.cuda().unsqueeze(0), indices)
        base.append(reference[0])
        features.append(torch.cat((inputs[0], context.cuda()), -1))
        labels.append(label)
    return torch.stack(base), torch.stack(features), torch.tensor(labels, device="cuda")


def train(a):
    start = perf_counter()
    cfg, fixture, recorded = read(a.config), read(a.fixture), read(a.fit_audit)
    assert recorded["fixture_sha256"] == digest(a.fixture) and recorded["config_sha256"] == digest(a.config)
    assert recorded["weights_sha256"] == a.baseline_sha256
    baseline = torch.load(a.reference_heads, weights_only=True, map_location="cpu")
    base, features, labels = load_inputs(a, fixture, recorded, cfg)
    arms = {}
    for name, keep in GROUPS.items():
        mask = [f in keep for f in range(features.shape[-1])]
        x = features * torch.tensor(mask, device="cuda")
        torch.manual_seed(51407)
        model = ContextScorer().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["pair_learning_rate"], fused=True)
        rng = torch.Generator(device="cuda").manual_seed(19113)
        for _ in range(cfg["pair_epochs"]):
            for batch in torch.randperm(len(base), device="cuda", generator=rng).split(cfg["batch"]):
                loss = cross_entropy(model(base[batch], x[batch]), labels[batch])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        assert torch.isfinite(loss)
        arms[name] = {"mask": mask, "weights": model.cpu().state_dict(), "loss": float(loss.detach())}
    max_delta = max(float((tensor - baseline["heads"]["context_only"][param]).abs().max())
                    for param, tensor in arms["refit_full"]["weights"].items())
    meta = {"baseline_sha256": a.baseline_sha256, "config_sha256": digest(a.config),
            "fixture_sha256": digest(a.fixture), "fit_audit_sha256": digest(a.fit_audit),
            "reference_heads_sha256": digest(a.reference_heads), "full_refit_max_param_delta": max_delta,
            "elapsed_seconds": perf_counter() - start, "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__)}
    torch.save({**meta, "arms": arms}, a.output)
    save(a.audit, {**meta, "arms": {k: {m: r[m] for m in ("mask", "loss")} for k, r in arms.items()}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("inputs", "assets", "config", "fixture", "fit-audit",
                "baseline", "reference-heads", "output", "audit"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    train(p.parse_args())


if __name__ == "__main__":
    main()
