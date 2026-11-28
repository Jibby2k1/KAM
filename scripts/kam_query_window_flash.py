#!/usr/bin/env python3
"""Use fused causal attention on a small query window in each protected layer."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import BackboneAttention, WIDTHS, load
from eval_kam_long_sparse import language_score, retrieval_score
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_backbone_block_attention import HEAD_DIM, block_attention
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json

MODES = ("tail-32", "tail-256", "before-tail-256")
RUNS = tuple((seed, mode) for seed in SEEDS for mode in MODES)


def query_window(mode, width):
    if mode == "tail-32":
        return width-32, width
    if mode == "tail-256":
        return width-256, width
    if mode == "before-tail-256":
        return width-512, width-256
    if mode == "protected":
        return None
    raise ValueError(mode)


class WindowAttention(BackboneAttention):
    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        if self.mode == "flash":
            return self.original(module, q, k, v, mask, dropout=dropout, scaling=scaling, **kwargs)
        if mask is not None or dropout or not module.is_causal or q.size(-2) != k.size(-2):
            raise ValueError("Query-window experiment requires unpadded, uncached causal prefill")
        window = query_window(self.mode, q.size(-2))
        return block_attention(q, k, v, scaling, keep_first=True, flash_queries=window), None


@torch.inference_mode()
def verify():
    torch.manual_seed(731251)
    width = 1024

    def sample(h):
        return torch.randn(1, width, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)

    q, k, v = sample(32), sample(8), sample(8)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).transpose(1, 2)
        sparse = block_attention(q, k, v, keep_first=True)
        results = {}
        for mode in MODES:
            start, stop = query_window(mode, width)
            actual = block_attention(q, k, v, keep_first=True, flash_queries=(start, stop))
            torch.testing.assert_close(actual[:, start:stop], expected[:, start:stop], atol=.012, rtol=.04)
            assert torch.equal(actual[:, :start], sparse[:, :start])
            assert torch.equal(actual[:, stop:], sparse[:, stop:])
            results[mode] = float((actual[:, start:stop].float()-expected[:, start:stop].float()).abs().max())
    return results


def run(root, source, work, mode, index):
    if mode == "smoke":
        assert torch.cuda.get_device_name() == GPU
        print(verify(), flush=True)
        return
    seed, variant = RUNS[index] if mode == "evaluate" else (SEEDS[0], "flash")
    model, stack, projection, data, checkpoint, attention = load(root, source, work, seed, WindowAttention)
    common = {"seed": seed, "gpu": GPU, "checkpoint_sha256": checkpoint,
              "manifest_sha256": digest(work / "manifest.json")}
    if mode == "evaluate":
        attention.mode = variant
        with attention.installed():
            scores = {"pg19": language_score(model, stack, data),
                      "retrieval": retrieval_score(model, stack, projection, data)}
        write_json(work / f"window-{seed}-{variant}.json", common | {"variant": variant, **scores})
        return

    parity = verify()
    values = [token for b in data["pg19"][0]["windows"] for token in b[:-1]]
    timings = {}
    with attention.installed():
        for width in WIDTHS:
            tokens = torch.tensor([values[:width]], device="cuda")
            mask = torch.ones_like(tokens, dtype=torch.bool)

            def action(name):
                attention.mode = name
                return next_logits(model, stack, tokens, mask, projection=projection)[0]

            modes = ("flash", "protected", *MODES)
            for label in modes:
                assert torch.isfinite(action(label)).all()
            timings[width] = profile({label: lambda label=label: action(label) for label in modes}, repeats=15)
    write_json(work / "query-window-profile.json", common | {"parity": parity, "torch": str(torch.__version__),
                                                            "full_prefill": timings})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "work"):
        p.add_argument(name, type=Path)
    p.add_argument("mode", choices=("smoke", "evaluate", "benchmark"))
    p.add_argument("index", type=int, nargs="?", choices=range(len(RUNS)))
    a = p.parse_args()
    run(a.root, a.source, a.work, a.mode, a.index)
