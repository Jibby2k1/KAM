#!/usr/bin/env python3
"""Fit lightweight block-score corrections from frozen dense attention."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_kam_backbone_shortlist import BackboneAttention, load
from kam_adapter_preservation import SEEDS, digest
from kam_backbone_block_attention import BLOCK, BUDGET, GROUP, QUERY_TILE, block_routes
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_pretrained_adapters import mixed_precision
from prepare_kam_route_distillation import PROTOCOL, read

FEATURES = ("max_proxy", "mean_proxy", "inverse_distance", "inverse_block_index")
MODELS = {"moments": 2, "moments-position": 4}
RIDGE = .005
FLEX = BUDGET-3
STRIDE = 128


class RouteTeacher(BackboneAttention):
    def initialize(self, device):
        self.rows = torch.arange(512, 4096, STRIDE, device=device)
        self.positions = torch.arange(4096, device=device)
        self.blocks = torch.arange(4096//BLOCK, device=device)
        self.current = self.rows // BLOCK
        self.eligible = (self.blocks > 0) & (self.blocks < self.current[:, None]-1)
        self.fixed = (self.blocks == 0) | (self.blocks == self.current[:, None]) | (self.blocks == self.current[:, None]-1)
        self.distance = (self.current[:, None]-self.blocks).clamp_min(1).float().reciprocal()
        self.early = (self.blocks+1).float().reciprocal().expand_as(self.distance)
        self.visible = self.positions <= self.rows[:, None]
        self.covariance = torch.zeros(32, 32, len(FEATURES), len(FEATURES), device=device, dtype=torch.float64)
        self.target = torch.zeros(32, 32, len(FEATURES), device=device, dtype=torch.float64)

    def process(self, idx, q, k, scale):
        heads, kv, dim = q.size(1), k.size(1), k.size(-1)
        assert q.shape == (1, 32, 4096, 128) and heads % kv == 0
        with torch.autocast("cuda", enabled=False):
            query = q.index_select(-2, self.rows).float().reshape(1, kv, -1, dim)
            keys = k.float()
            exact = (query @ keys.transpose(-1, -2)).reshape(heads, len(self.rows), 4096) * scale
            logsumexp = exact.masked_fill(~self.visible, -torch.inf).unflatten(-1, (64, BLOCK)).logsumexp(-1)
            means = keys.unflatten(-2, (4096//GROUP, GROUP)).mean(-2).bfloat16().float()
            proxy = (query @ means.transpose(-1,-2)).reshape(heads, len(self.rows), 64, BLOCK//GROUP) * scale
            # Positional summaries share the same block and query axes as the projected scores.
            extra = [f.expand(heads, -1, -1) for f in (self.distance, self.early)]
            x = torch.stack((proxy.amax(-1), proxy.mean(-1), *extra), dim=-1)
            if self.phase == "fit":
                self.collect(idx, x, logsumexp)
            else:
                pages = keys.unflatten(-2, (64, BLOCK))
                upper, lower = pages.amax(-2), pages.amin(-2)
                bound = (query.clamp_min(0) @ upper.transpose(-1, -2) +
                         query.clamp_max(0) @ lower.transpose(-1, -2)).reshape(heads, len(self.rows), 64) * scale
                self.measure(idx, x, logsumexp, q, k, bound)

    def collect(self, idx, x, target):
        teacher = target.masked_fill(~self.eligible, -torch.inf).softmax(-1)
        weight = teacher + .05 * (self.eligible / self.eligible.sum(-1, keepdim=True))
        weight /= 1.05
        target = target.masked_fill(~self.eligible, 0)
        x = x - (x * weight[..., None]).sum(-2, keepdim=True)
        target = target - (target * weight).sum(-1, keepdim=True)
        centered = x.flatten(1, 2)
        weighted = (x * weight[..., None]).flatten(1, 2)
        self.covariance[idx] += centered.transpose(-1, -2) @ weighted
        self.target[idx] += (weighted.transpose(-1, -2) @ target.flatten(1).unsqueeze(-1)).squeeze(-1)

    def fit(self, device):
        c = self.covariance.cpu().numpy()
        y = self.target.cpu().numpy()
        self.coefficients = {}
        for name, n in MODELS.items():
            matrix = c[..., :n, :n]
            ridge = RIDGE * np.maximum(np.trace(matrix, axis1=-2, axis2=-1) / n, 1e-9)
            prior = np.zeros(n)
            prior[0] = 1
            fit = np.linalg.solve(matrix + ridge[..., None, None]*np.eye(n),
                                  (y[..., :n]+ridge[..., None]*prior)[..., None]).squeeze(-1)
            self.coefficients[name] = torch.tensor(fit, device=device, dtype=torch.float32)

    def measure(self, idx, x, logsumexp, q, k, bound):
        probs = logsumexp.softmax(dim=-1)
        past = probs * self.eligible
        fixed = (probs * self.fixed).sum(-1)
        routed = block_routes(q, k, keep_first=True)[0].index_select(-2, self.rows//QUERY_TILE).long()
        actual = probs.gather(-1, routed).sum(-1)
        # Correct only the ranking among blocks that are not already mandatory.
        scores = {"oracle": logsumexp, "minmax-bound": bound}
        for name, coeffs in self.coefficients.items():
            width = coeffs.size(-1)
            scores[name] = (x[..., :width] * coeffs[idx, :, None, None, :]).sum(-1)
        candidates = {"protected": actual}
        for name, rank in scores.items():
            picked = rank.masked_fill(~self.eligible, -torch.inf).argsort(dim=-1, descending=True, stable=True)[..., :FLEX]
            candidates[name] = fixed + probs.gather(-1, picked).sum(-1)
        totals = {"fixed": fixed.mean(dim=-1), "adjustable_mass": past.sum(-1).mean(dim=-1),
                  **{name: val.mean(dim=-1) for name, val in candidates.items()}}
        for name, value in totals.items():
            self.metrics[name][idx] += value

    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        assert mask is None and dropout == 0 and not self.selected and scaling is not None
        self.process(module.layer_idx, q, k, scaling)
        return super().forward(module, q, k, v, mask, dropout=dropout, scaling=scaling, **kwargs)


@torch.inference_mode()
def run(root, source, work, index):
    seed = SEEDS[index]
    manifest = work/"routing-input-manifest.json"
    panel = work/"routing-inputs.json"
    spec = read(manifest)
    assert spec["protocol"] == PROTOCOL and spec["inputs_sha256"] == digest(panel)
    corpus = read(panel)
    for group, books in corpus.items():
        assert [b["id"] for b in books] == [b["id"] for b in spec["groups"][group]]
    assert not ({r["id"] for r in corpus["fit"]} & {r["id"] for r in corpus["dev"]})
    model, stack, _, _, checkpoint, teacher = load(root, source, work, seed, RouteTeacher)
    teacher.initialize("cuda")
    score_names = ("fixed", "adjustable_mass", "protected", *MODELS, "minmax-bound", "oracle")

    def run_book(book):
        for window in book["windows"]:
            model.model(input_ids=torch.tensor([window], device="cuda"), use_cache=False)

    with teacher.installed(), mixed_precision():
        stack.context(None, False)
        teacher.phase = "fit"
        for book in corpus["fit"]:
            run_book(book)
        teacher.fit("cuda")
        teacher.phase = "dev"
        results = []
        for book in corpus["dev"]:
            teacher.metrics = {key: torch.zeros((32, 32), device="cuda", dtype=torch.float64) for key in score_names}
            run_book(book)
            n = len(book["windows"])
            results.append({"id": book["id"], "windows": n,
                "mean_mass": {key: (values/n).tolist() for key, values in teacher.metrics.items()}})
    metadata = {"gpu": GPU, "seed": seed, "checkpoint_sha256": checkpoint, "input_manifest_sha256": digest(manifest),
                "feature_order": FEATURES, "ridge": RIDGE, "query_stride": STRIDE, "flex_slots": FLEX}
    write_json(work/f"route-fit-{seed}.json", metadata | {"fit_books": [b["id"] for b in corpus["fit"]],
        "coefficients": {name: t.tolist() for name, t in teacher.coefficients.items()}, "dev": results})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "work"):
        p.add_argument(name,type=Path)
    p.add_argument("index",type=int,choices=range(len(SEEDS)))
    a=p.parse_args()
    run(a.root, a.source, a.work, a.index)
