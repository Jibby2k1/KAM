#!/usr/bin/env python3
"""Audit exact-key recovery and test shortlisted top-four on the frozen holdout."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark_kam_sparse_live_read import dense, profile, reference, tensors
from eval_kam_long_sparse import InferenceAdapter, InferenceStack, language_score, prepare_run, retrieval_score
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_block_shortlist import BLOCK, BUDGETS, PROXIES, block_shortlist_read
from kam_exact_sparse_readout import exact_sparse_read
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_pretrained_adapters import HEADS, WIDTH

RUNS = tuple((seed, budget) for seed in SEEDS for budget in BUDGETS)
PRIMARY = 8
LENGTHS = (512, 2048, 4096, 8192)
ARM = "top4-linear"


def exact_indices(q, k, rows):
    with torch.autocast(q.device.type, enabled=False):
        score = (q.index_select(-2, rows).float() @ k.float().transpose(-2, -1)) * .25
    stop = rows + k.size(-2) - q.size(-2)
    score = score.masked_fill(torch.arange(k.size(-2), device=k.device) > stop[:, None], -torch.inf)
    indices = score.argsort(dim=-1, descending=True, stable=True)[..., :min(4, k.size(-2))]
    return indices.masked_fill(indices > stop[:, None], -1)


def audit_read(q, k, ids):
    rows = torch.arange(31, q.size(-2), 64, device=q.device)
    expected = exact_indices(q, k, rows)
    actual = ids.index_select(-2, rows)
    hit = ((actual[..., :, None] == expected[..., None, :]).any(dim=-2) & (expected >= 0))
    eligible = (expected >= 0).sum(dim=-1)
    result = {}
    for label, low, high in (("first_quarter", 0, 1), ("middle_half", 1, 3), ("last_quarter", 3, 4)):
        region = (4 * rows >= q.size(-2) * low) & (4 * rows < q.size(-2) * high)
        good = hit[..., region, :].sum(dim=-1)
        count = eligible[..., region]
        result[label] = (int(good.sum()), int(count.sum()), int((good == count).sum()), count.numel())
    return result


class ShortlistAdapter(InferenceAdapter):
    def forward(self, x):
        if self.backend != "shortlist":
            return super().forward(x)
        if self.cache_enabled or self.cache is not None:
            raise ValueError("The full-reader benchmark measures uncached causal prefill")
        z = self.down(F.rms_norm(x, (x.shape[-1],)))
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        q = q + self.query_out(self.query_in(z))
        q, k, v = (t.unflatten(-1, (HEADS, WIDTH // HEADS)).transpose(1, 2) for t in (q, k, v))
        if self.audit is None:
            value = block_shortlist_read(q, k, v, self.budget)
        else:
            value, ids = block_shortlist_read(q, k, v, self.budget, record=True)
            for region, counts in audit_read(q, k, ids).items():
                for i, count in enumerate(counts):
                    self.audit[region][i] += count
        value = self.out(value.transpose(1, 2).reshape_as(z))
        return x + self.up(value).to(x.dtype)


class ShortlistStack(InferenceStack):
    adapter_type = ShortlistAdapter

    def configure(self, budget, audit=None):
        for a in self.modules:
            a.budget = budget
            a.audit = audit
            a.backend = "shortlist"


def load(root, source, work, seed):
    return prepare_run(root, source, work, seed, ARM, stack_factory=ShortlistStack)


def evaluate(root, source, work, index):
    seed, budget = RUNS[index]
    model, stack, proj, data, checkpoint = load(root, source, work, seed)
    audit = {}
    # Four books and four mirrored pairs, locked by their positions in the prior manifest.
    for domain, settings in (("pg19", dict(data={**data, "pg19": [
            {**r, "windows": r["windows"][:1]} for r in data["pg19"][:4]]})),
            ("musique", dict(data=data, limit=4))):
        counts = {r: [0, 0, 0, 0] for r in ("first_quarter", "middle_half", "last_quarter")}
        stack.configure(budget, counts)
        if domain == "pg19":
            language_score(model, stack, **settings)
        else:
            retrieval_score(model, stack, proj, **settings)
        audit[domain] = counts
    stack.configure(budget)
    payload = {"pg19": language_score(model, stack, data), "retrieval": retrieval_score(model, stack, proj, data)}
    write_json(work / f"block-{seed}-{budget}.json", {"gpu": GPU, "seed": seed, "budget": budget,
        "manifest_sha256": digest(work / "manifest.json"), "checkpoint_sha256": checkpoint,
        "key_recovery": audit, "layers": len(stack.modules), **payload})


@torch.inference_mode()
def fidelity():
    trials = {}
    torch.manual_seed(731207)
    for b, nq, nk in ((1, 1, 1), (2, 97, 173), (1, 257, 257), (1, 1, 512), (1, 512, 512)):
        q, k, v = tensors(b, nq, nk)
        budget = 8
        out, ids = block_shortlist_read(q, k, v, budget, record=True)
        torch.testing.assert_close(out, reference(q, k, v, True), rtol=.01, atol=.016)
        rows = torch.arange(nq, device=q.device)
        expected = exact_indices(q, k, rows)
        assert torch.equal(ids[..., :expected.size(-1)], expected)
        trials[f"{b}-{nq}-{nk}"] = {"ids": int((expected >= 0).sum())}
    q, k, v = tensors(1, 640, 640)
    out, ids = block_shortlist_read(q, k, v, 4, record=True)
    k[..., 319:, :] = v[..., 319:, :] = 100
    altered, new_ids = block_shortlist_read(q, k, v, 4, record=True)
    assert torch.equal(ids[..., :319, :], new_ids[..., :319, :])
    assert torch.equal(out[..., :319, :], altered[..., :319, :])
    return trials


@torch.inference_mode()
def benchmark(root, source, work):
    import triton

    assert torch.cuda.get_device_name() == GPU
    checks = fidelity()
    kernel = {}
    for nk in LENGTHS:
        for kind, nq in (("prefill", nk), ("read-one", 1)):
            q, k, v = tensors(1, nq, nk)
            kernel[f"{kind}-{nk}"] = profile({"flash": lambda: dense(q, k, v, True),
                "exact": lambda: exact_sparse_read(q, k, v, causal=True),
                **{str(b): lambda b=b: block_shortlist_read(q, k, v, b) for b in BUDGETS}})
    seed = SEEDS[0]
    model, stack, proj, data, checkpoint = load(root, source, work, seed)
    sample = [token for seq in data["pg19"][0]["windows"] for token in seq[:-1]]
    full = {}
    for width in LENGTHS:
        tokens = torch.tensor([sample[:width]], device="cuda")
        mask = torch.ones_like(tokens, dtype=torch.bool)

        def action(mode):
            if isinstance(mode, int):
                stack.configure(mode)
            else:
                stack.backend(mode)
            return next_logits(model, stack, tokens, mask, projection=proj)[0]

        modes = ("flash", "sparse", *BUDGETS)
        full[str(width)] = profile({("exact" if m == "sparse" else str(m)): lambda m=m: action(m)
                                    for m in modes}, repeats=15)
    write_json(work / "block-profile.json", {"gpu": GPU, "torch": str(torch.__version__), "triton": triton.__version__,
        "seed": seed, "checkpoint_sha256": checkpoint, "manifest_sha256": digest(work / "manifest.json"),
        "block": BLOCK, "proxies": PROXIES, "primary_budget": PRIMARY, "checks": checks,
        "kernel": kernel, "full_prefill": full})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "work"):
        p.add_argument(name, type=Path)
    p.add_argument("mode", choices=("smoke", "benchmark", "evaluate"))
    p.add_argument("index", type=int, nargs="?", choices=range(len(RUNS)))
    a = p.parse_args()
    if a.mode == "smoke":
        assert torch.cuda.get_device_name() == GPU
        print(fidelity())
    elif a.mode == "benchmark":
        benchmark(a.root, a.source, a.work)
    else:
        evaluate(a.root, a.source, a.work, a.index)
