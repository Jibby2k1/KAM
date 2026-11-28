#!/usr/bin/env python3
"""Calibrate static layer budgets from independent frozen Flash activations."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import BackboneAttention, load
from eval_kam_long_sparse import language_score, retrieval_score
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_backbone_block_attention import BLOCK, HEAD_DIM, QUERY_TILE, block_attention, block_routes
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_pretrained_adapters import mixed_precision
from kam_query_window_flash import WindowAttention

BUDGETS = (4, 8, 12)
POLICIES = ("output-selected", "evidence-selected", "alternating")
WIDTHS = (4096, 8192)
RUNS = tuple((s, arm) for s in SEEDS for arm in POLICIES)
PROTOCOL = "kam-layer-budget-v1"


def read(path):
    return json.loads(path.read_text())


class RoutedTeacher(BackboneAttention):
    def record(self, module, q, k, v, scaling):
        assert scaling is not None and q.shape == (1, 32, 4096, 128)
        heads, kv, width, dim = q.size(1), k.size(1), q.size(2), q.size(3)
        rows = self.rows
        with torch.autocast("cuda", enabled=False):
            query = q[0].index_select(-2, rows).float().reshape(kv, -1, dim)
            keys, values = k[0].float(), v[0].float()
            scores = ((query @ keys.transpose(-1, -2)) * scaling).reshape(heads, len(rows), width)
            scores = scores.masked_fill(self.columns > rows[:, None], -torch.inf)
            weights = scores.softmax(-1)
            dense = (weights.flatten(1, 2).view(kv, -1, width) @ values).view(heads, len(rows), dim)
            dense = dense.transpose(0, 1).reshape(len(rows), -1)
            o = module.o_proj.weight.float()
            norm = F.linear(dense, o).square().sum()
            idx = module.layer_idx
            self.agg["output_energy"][idx] += norm
            if self.span is not None:
                start, stop = self.span
                evidence = (self.columns >= max(start, BLOCK)) & (self.columns < stop)
                self.agg["dense_evidence"][idx] += (weights * evidence).sum() / heads
            for budget in BUDGETS:
                route = block_routes(q, k, keep_first=True, budget=budget)[0].index_select(1, rows // QUERY_TILE)
                used = torch.arange(budget, device=q.device) < rows[:, None] // BLOCK + 1
                keep = ((route[..., None] == self.key_blocks) & used[..., None]).any(dim=-2)
                weights_sparse = scores.masked_fill(~keep, -torch.inf).softmax(-1)
                sparse = (weights_sparse.flatten(1, 2).view(kv, -1, width) @ values).view(heads, len(rows), dim)
                error = sparse.transpose(0, 1).reshape(len(rows), -1) - dense
                label = str(budget)
                self.agg["output_sse"][label][idx] += F.linear(error, o).square().sum()
                self.agg["retained_mass"][label][idx] += (weights * keep).sum() / heads
                if self.span is not None:
                    self.agg["retained_evidence"][label][idx] += (weights * keep * evidence).sum() / heads

    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        assert mask is None and dropout == 0 and not self.selected
        self.record(module, q, k, v, scaling)
        return super().forward(module, q, k, v, mask, dropout=dropout, scaling=scaling, **kwargs)

    def reset(self):
        def row():
            return torch.zeros(32, dtype=torch.float64, device="cuda")
        self.agg = {key: {str(n): row() for n in BUDGETS}
                    for key in ("output_sse", "retained_mass", "retained_evidence")}
        self.agg |= {name: row() for name in ("output_energy", "dense_evidence")}

    def scores(self):
        return {k: ({n: v.tolist() for n, v in value.items()} if isinstance(value, dict) else value.tolist())
                for k, value in self.agg.items()}


class ScheduledAttention(WindowAttention):
    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        if self.mode not in POLICIES:
            return super().forward(module, q, k, v, mask, dropout, scaling, **kwargs)
        if mask is not None or dropout or not module.is_causal or q.size(-2) != k.size(-2):
            raise ValueError("Budget schedules require strictly causal unpadded prefill")
        budget = self.schedule[self.mode][module.layer_idx]
        return block_attention(q, k, v, scaling, keep_first=True, budget=budget), None


@torch.inference_mode()
def smoke():
    torch.manual_seed(734841)
    width, heads, kv = 1024, 8, 2

    def sample(h):
        return torch.randn(2, width, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)

    q, k, v = sample(heads), sample(kv), sample(kv)
    positions = torch.arange(width, device=q.device)
    results = {}
    for budget in BUDGETS:
        actual, route = block_attention(q, k, v, keep_first=True, budget=budget, record_routes=True)
        current = positions // BLOCK
        used = torch.arange(budget, device=q.device) < current[:, None] + 1
        picked = route.index_select(-2, positions // QUERY_TILE).masked_fill(~used, -1)
        assert ((picked == 0).any(dim=-1)).all() and (picked <= current[:, None]).all()
        assert not ((picked[..., :, None] == picked[..., None, :]).tril(-1) & used[..., :, None] & used[..., None, :]).any()
        visible = (picked[..., None] == current).any(dim=-2) & (positions <= positions[:, None])
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=visible,
                                                      enable_gqa=True).transpose(1, 2).bfloat16()
        torch.testing.assert_close(actual, expected, atol=.012, rtol=.04)
        results[str(budget)] = float((actual.float() - expected.float()).abs().max())
    return results


@torch.inference_mode()
def calibrate(root, source, work, index):
    seed = SEEDS[index]
    model, stack, projection, _, checkpoint, teacher = load(root, source, work, seed, RoutedTeacher)
    spans_path = work / "spans.json"
    spans = read(spans_path)
    qa_path = work / "dev-inputs.json"
    assert spans["dev_inputs_sha256"] == digest(qa_path)
    questions = read(qa_path)
    assert len(questions) == len(spans["dev"]) == 96
    manifest_path = work / "routing-input-manifest.json"
    corpus_path = work / "routing-inputs.json"
    assert read(manifest_path)["inputs_sha256"] == digest(corpus_path)
    books = read(corpus_path)["dev"]
    teacher.columns = torch.arange(4096, device="cuda")
    teacher.key_blocks = teacher.columns // BLOCK
    output = {}
    with teacher.installed(), mixed_precision():
        stack.context(None, False)
        teacher.rows = torch.arange(511, 4096, 256, device="cuda")
        teacher.span = None
        records = []
        n_sampled = len(teacher.rows)
        for book in books:
            teacher.reset()
            for window in book["windows"]:
                model.model(input_ids=torch.tensor([window], device="cuda"), use_cache=False)
            records.append({"id": book["id"], "n_queries": len(book["windows"]) * n_sampled, "values": teacher.scores()})
        output["books"] = records
        teacher.rows = torch.tensor([4095], device="cuda")
        records = []
        for start in range(0, len(questions), 2):
            teacher.reset()
            for row, s in zip(questions[start:start+2], spans["dev"][start:start+2], strict=True):
                assert (row["id"], row["expected"], row["family"]) == (s["id"], s["expected"], s["family"])
                teacher.span = s["earlier"]
                model.model(input_ids=torch.tensor([row["ids"]], device="cuda"), use_cache=False)
            records.append({"id": row["id"], "n_queries": 2, "values": teacher.scores()})
        output["musique"] = records
    write_json(work / f"layer-calibration-{seed}.json", {"protocol": PROTOCOL, "seed": seed, "gpu": GPU,
        "checkpoint_sha256": checkpoint, "spans_sha256": digest(spans_path), "pg_manifest_sha256": digest(manifest_path),
        "results": output})


@torch.inference_mode()
def evaluate(root, source, work, mode, index):
    seed, arm = RUNS[index] if mode == "evaluate" else (SEEDS[0], mode)
    plan_path = work / "layer-budget-selection.json"
    plan = read(plan_path)
    assert plan["protocol"] == PROTOCOL and set(plan["schedule"]) == set(POLICIES)
    assert all(s.count(4) == s.count(12) == 16 for s in plan["schedule"].values())
    model, stack, projection, _, checkpoint, attention = load(root, source, work, seed, ScheduledAttention)
    assert plan["checkpoints"][str(seed)] == checkpoint
    attention.schedule = plan["schedule"]
    transfer_path = work / "transfer-manifest.json"
    hot_path = work / "hotpot-manifest.json"
    outcomes = {}
    with attention.installed():
        for width in WIDTHS:
            m, h = work / f"transfer-{width}.json", work / f"hotpot-{width}.json"
            assert [digest(a) for a in (m,h)] == [read(a)["fixture"][str(width)]["sha256"] for a in (transfer_path,hot_path)]
            fixture = read(m)
            if mode == "evaluate":
                attention.mode = arm
                outcomes[width] = {"pg19": language_score(model, stack, fixture, context=width),
                    "retrieval": retrieval_score(model, stack, projection, fixture, context=width),
                    "hotpot": retrieval_score(model, stack, projection, {"retrieval": read(h)}, context=width)}
            else:
                tokens = torch.tensor([fixture["retrieval"][0]["ids"]], device="cuda")
                mask = torch.ones_like(tokens, dtype=torch.bool)

                def action(policy):
                    attention.mode = policy
                    return next_logits(model, stack, tokens, mask, projection=projection)[0]

                arms = ("flash", "protected", *POLICIES)
                outcomes[width] = profile({a: lambda a=a: action(a) for a in arms}, repeats=15)
    name = f"layer-budget-{seed}-{arm}.json" if mode == "evaluate" else "layer-budget-profile.json"
    write_json(work/name, {"protocol": PROTOCOL, "seed": seed, "gpu": GPU, "arm": arm, "checkpoint_sha256": checkpoint,
        "plan_sha256": digest(plan_path), "musique_manifest_sha256": digest(transfer_path),
        "hotpot_manifest_sha256": digest(hot_path), "outcomes": outcomes})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "source", "work"):
        cli.add_argument(key, type=Path)
    cli.add_argument("mode", choices=("smoke", "calibrate", "benchmark", "evaluate"))
    cli.add_argument("index", nargs="?", type=int, choices=range(len(RUNS)))
    a = cli.parse_args()
    if a.mode == "smoke":
        assert torch.cuda.get_device_name() == GPU
        write_json(a.work/"layer-budget-smoke.json", smoke())
    elif a.mode == "calibrate":
        calibrate(a.root, a.source, a.work, a.index)
    else:
        evaluate(a.root, a.source, a.work, a.mode, a.index)
