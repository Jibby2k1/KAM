#!/usr/bin/env python3
"""Run differentiable attention over the original discrete Triton routes."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from benchmark_kam_sparse_live_read import profile
from kam_backbone_block_attention import BLOCK, BUDGET, HEAD_DIM, QUERY_TILE, block_attention, block_routes

FLEX_TILE = 128
KERNEL_OPTIONS = {"fwd_BLOCK_M": 32, "fwd_BLOCK_N": 32, "fwd_num_stages": 1,
                  "bwd_BLOCK_M1": 32, "bwd_BLOCK_N1": 32, "bwd_BLOCK_M2": 32,
                  "bwd_BLOCK_N2": 32, "bwd_num_stages": 1}
compiled_flex = torch.compile(flex_attention, dynamic=False, fullgraph=True)


def make_mask(route, width):
    assert width % FLEX_TILE == 0 and route.size(-2) == width // QUERY_TILE
    num_blocks = width // BLOCK
    tiles = torch.arange(route.size(-2), device=route.device) * QUERY_TILE // BLOCK
    used = torch.arange(route.size(-1), device=route.device) < tiles[:, None] + 1
    chosen = torch.where(used, route, num_blocks).long()
    allowed = torch.zeros((*route.shape[:-1], num_blocks+1), dtype=torch.bool, device=route.device)
    allowed.scatter_(-1, chosen, True)
    allowed = allowed[..., :num_blocks].contiguous()
    # A larger FlexAttention tile is active if any constituent original route touches it.
    # The per-token predicate still enforces the original route and causal boundary.
    count = width // FLEX_TILE
    coarse = allowed.reshape(*route.shape[:2], count, FLEX_TILE//QUERY_TILE, count, FLEX_TILE//BLOCK).any(dim=(3,5))
    indices = coarse.int().argsort(dim=-1, descending=True, stable=True).int()

    def exact(batch, head, row, column):
        return (row >= column) & allowed[batch, head, row//QUERY_TILE, column//BLOCK]

    return BlockMask.from_kv_blocks(coarse.sum(dim=-1, dtype=torch.int32).contiguous(), indices,
                                    BLOCK_SIZE=FLEX_TILE, mask_mod=exact, seq_lengths=(width,width))


def differentiable_sparse(q, k, v, scale, *, route=None):
    if route is None:
        with torch.no_grad():
            route = block_routes(q, k, keep_first=True)
    return compiled_flex(q, k, v, scale=scale, block_mask=make_mask(route, q.size(-2)),
                         enable_gqa=q.size(1) != k.size(1), kernel_options=KERNEL_OPTIONS)


def smoke(output):
    torch.manual_seed(735019)
    checks = {}
    for batch, width, heads, kv in ((1, 1024, 8, 2), (1, 4096, 32, 8)):
        def sample(n):
            return torch.randn(batch, width, n, HEAD_DIM, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        q, k, v = (sample(n).requires_grad_(True) for n in (heads, kv, kv))
        with torch.no_grad():
            triton, route = block_attention(q, k, v, keep_first=True, record_routes=True)
        scale = HEAD_DIM ** -.5
        observed = differentiable_sparse(q, k, v, scale, route=route)
        torch.testing.assert_close(triton.transpose(1, 2), observed, atol=.016, rtol=.045)
        dv = sample(heads)
        grad = torch.autograd.grad(observed, (q, k, v), dv)
        assert all(torch.isfinite(x).all() and x.norm().item()>0 for x in grad)
        ref_diff = {}
        if width == 1024:
            positions = torch.arange(width, device=q.device)
            take = torch.arange(BUDGET, device=q.device) < positions[:, None] // BLOCK + 1
            chosen = route.index_select(-2, positions // QUERY_TILE).masked_fill(~take, -1)
            valid = (chosen[..., None] == positions // BLOCK).any(dim=-2) & (positions <= positions[:, None])
            with sdpa_kernel(SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=valid,
                                                        enable_gqa=True, scale=scale)
            ref = torch.autograd.grad(expected, (q, k, v), dv.float())
            for label, a, b in zip(("q","k","v"), grad, ref, strict=True):
                relative = (a.float()-b.float()).norm() / b.float().norm()
                assert relative < .02, (label, relative.item())
                ref_diff[label] = float(relative)
        def action():
            value = differentiable_sparse(q, k, v, scale)
            return torch.autograd.grad(value, (q,k,v), dv)
        timings = profile({"forward-backward": action}, repeats=12)
        checks[str(width)] = {"relative_gradient_error": ref_diff, "timings": timings,
                             "torch_shape": [batch, heads, kv, width, HEAD_DIM],
                             "max_abs_forward_delta": float((triton.transpose(1, 2).float()-observed.float()).abs().max())}
    output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(),"torch":str(torch.__version__),
                                  "flex_tile":FLEX_TILE, "kernel_options":KERNEL_OPTIONS, "checks":checks}, sort_keys=True)+"\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    smoke(args.output)
