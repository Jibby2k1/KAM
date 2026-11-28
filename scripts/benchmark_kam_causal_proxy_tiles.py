#!/usr/bin/env python3
"""Measure causal proxy tiling with unchanged shortlist decisions."""
import argparse
import json
from pathlib import Path

import torch

from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import load
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_backbone_block_attention import HEAD_DIM, block_attention, block_routes
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_query_window_flash import WindowAttention, query_window

PROTOCOL = "kam-causal-proxy-tiling-v1"
ROUTES = {"rectangular": (False, False), "causal-tiles": (True, False), "causal-local": (True, True)}
SHAPES = ((1, 4096), (4, 4096), (1, 8192), (4, 8192))


def read(path):
    return json.loads(path.read_text())


class ProxyAttention(WindowAttention):
    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        if self.mode in ("flash", "protected", "tail-256"):
            return super().forward(module, q, k, v, mask, dropout, scaling, **kwargs)
        if mask is not None or dropout or not module.is_causal or q.size(-2) != k.size(-2):
            raise ValueError("Proxy tiles require strictly causal unpadded prefill")
        route, layout = self.mode.split(":", 1)
        tiled, local = ROUTES[route]
        return block_attention(q, k, v, scaling, keep_first=True, flash_queries=query_window(layout, q.size(-2)),
                               bucket_routes=tiled, local_proxy_tile=local), None


def samples(batch, width, heads=32, kv=8):
    def sample(n):
        return torch.randn(batch, width, n, HEAD_DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    return sample(heads), sample(kv), sample(kv)


@torch.inference_mode()
def smoke():
    torch.manual_seed(928510)
    checks = {}
    for batch, width in ((1, 103), (2, 1024), (1, 2749), *SHAPES):
        q, k, v = samples(batch, width)
        first, route = block_attention(q, k, v, keep_first=True, record_routes=True)
        tests = {}
        for tile in ("causal-tiles", "causal-local"):
            tiled, local = ROUTES[tile]
            second, alt = block_attention(q, k, v, keep_first=True, record_routes=True, bucket_routes=tiled,
                                          local_proxy_tile=local)
            tests[tile] = {"whole_route_equal": torch.equal(route, alt), "output_equal": torch.equal(first, second),
                           "route_elements": route.numel(), "changed_route_elements": int((route != alt).sum()),
                           "max_abs_output_delta": float((first.float() - second.float()).abs().max())}
            assert tests[tile]["whole_route_equal"] and tests[tile]["output_equal"]
        checks[f"{batch}x{width}"] = tests
    return checks


@torch.inference_mode()
def kernel_benchmark(work):
    parity = smoke()
    by_shape = {}
    for batch, width in SHAPES:
        q, k, v = samples(batch, width)

        def routes(name):
            bucket, local = ROUTES[name]
            return block_routes(q, k, keep_first=True, bucket_routes=bucket, local_proxy_tile=local)

        def full(name):
            bucket, local = ROUTES[name]
            return block_attention(q, k, v, keep_first=True, bucket_routes=bucket, local_proxy_tile=local)

        by_shape[f"{batch}x{width}"] = {"means_and_routes": profile({n: lambda n=n: routes(n) for n in ROUTES}),
                                        "whole_sparse_attention": profile({n: lambda n=n: full(n) for n in ROUTES})}
    name = torch.cuda.get_device_name()
    write_json(work/"proxy-kernel.json", {"protocol": PROTOCOL, "gpu": name, "torch": str(torch.__version__),
                                         "parity": parity, "timings": by_shape})


@torch.inference_mode()
def profile_model(root, source, work):
    seed = SEEDS[0]
    model, stack, projection, _, checkpoint, attention = load(root, source, work, seed, ProxyAttention)
    widths = (4096, 8192)
    fixture_path = work/"transfer-manifest.json"
    fixture = read(fixture_path)
    parity, timings = {}, {}
    plain = ("protected", "tail-256")
    modes = ("flash", *plain, *(f"{route}:{base}" for route in ("causal-tiles", "causal-local") for base in plain))
    datasets = {}
    for width in widths:
        path = work / f"transfer-{width}.json"
        assert digest(path) == fixture["fixture"][str(width)]["sha256"]
        datasets[width] = read(path)
    with attention.installed():
        for width in widths:
            prompts = datasets[width]["retrieval"][:8]
            tokens = torch.tensor([p["ids"] for p in prompts], device="cuda")
            mask = torch.ones_like(tokens, dtype=torch.bool)
            compared = {}
            for a in modes[1:]:
                attention.mode = a
                compared[a] = torch.cat([next_logits(model, stack, x, y, projection=projection)[0]
                                         for x, y in zip(tokens.split(2), mask.split(2), strict=True)], dim=0)
            result = {}
            for a in modes[3:]:
                reference = compared[a.split(":")[-1]]
                result[a] = {"exact": torch.equal(compared[a], reference), "values": reference.numel(),
                             "max_abs_delta": float((compared[a]-reference).abs().max())}
                assert result[a]["exact"]
            parity[str(width)] = result

        for batch, width in SHAPES:
            examples = datasets[width]["pg19"][:batch]
            tokens = torch.tensor([r["windows"][0][:-1] for r in examples], device="cuda")
            mask = torch.ones_like(tokens, dtype=torch.bool)
            def prefill(a):
                attention.mode = a
                return next_logits(model, stack, tokens, mask, projection=projection)[0]
            timings[f"{batch}x{width}"] = profile({a: lambda a=a: prefill(a) for a in modes}, repeats=15)
    write_json(work/"proxy-model.json", {"protocol": PROTOCOL, "gpu": GPU, "seed": seed,
        "checkpoint_sha256": checkpoint, "manifest_sha256": digest(fixture_path), "parity": parity, "timings": timings})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "source", "work"):
        cli.add_argument(key, type=Path)
    cli.add_argument("mode", choices=("kernel", "model"))
    args = cli.parse_args()
    if args.mode == "kernel":
        kernel_benchmark(args.work)
    else:
        profile_model(args.root, args.source, args.work)
