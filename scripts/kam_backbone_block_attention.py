"""Block-shortlisted softmax attention for unpadded BF16 Llama prefill."""
import math

import torch
import triton
import triton.language as tl

BLOCK = 64
GROUP = 16
QUERY_TILE = 32
BUDGET = 8
HEAD_DIM = 128


@triton.jit
def _mean_keys(K, Means, N: tl.constexpr, M: tl.constexpr, H: tl.constexpr,
               KB: tl.constexpr, KH: tl.constexpr, KT: tl.constexpr, G: tl.constexpr, D: tl.constexpr):
    group, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = group * G + tl.arange(0, G)
    dims = tl.arange(0, D)
    k = tl.load(K + batch * KB + head * KH + rows[:, None] * KT + dims[None, :], rows[:, None] < N, 0)
    mean = tl.sum(k.to(tl.float32), axis=0) / tl.minimum(G, N - group * G)
    tl.store(Means + ((batch * H + head) * M + group) * D + dims, mean)


@triton.jit
def _routes(Q, Means, Route, N: tl.constexpr, M: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
            KV: tl.constexpr, QB: tl.constexpr, QH: tl.constexpr, QT: tl.constexpr, R: tl.constexpr,
            B: tl.constexpr, SR: tl.constexpr, TS: tl.constexpr, SB: tl.constexpr,
            D: tl.constexpr, DOT_DIM: tl.constexpr, KEEP_FIRST: tl.constexpr):
    group, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tiles = group * R + tl.arange(0, R)
    rows = tiles * TS
    dim = tl.arange(0, DOT_DIM)
    proxy = tl.arange(0, SB)
    blocks = tl.arange(0, SB // SR)
    kv = head // (H // KV)
    coarse_scores = tl.full((R, SB), 0, tl.float32)
    for start in range(D // DOT_DIM):
        ch = start * DOT_DIM + dim
        q = tl.load(Q + batch * QB + head * QH + rows[:, None] * QT + ch[None, :], rows[:, None] < N, 0)
        means = tl.load(Means + ((batch * KV + kv) * M + proxy[None, :]) * D + ch[:, None], proxy[None, :] < M, 0)
        coarse_scores = tl.dot(q, means, coarse_scores)
    coarse = tl.max(tl.reshape(coarse_scores, (R, SB // SR, SR)), axis=2)
    position = rows // B
    # Score the current and previous blocks directly. All routed proxies precede both.
    coarse = tl.where(blocks[None, :] < position[:, None] - 1, coarse, float("-inf"))
    output = Route + ((batch * H + head) * T + tiles) * 8
    tl.store(output, position, tiles < T)
    tl.store(output + 1, position - 1, tiles < T)
    has_first = position < 2
    for rank in tl.static_range(2, 8):
        loc = tl.argmax(coarse, axis=1, tie_break_left=True)
        if KEEP_FIRST and rank == 7:
            loc = tl.where(has_first, loc, 0)
        tl.store(output + rank, loc, tiles < T)
        coarse = tl.where(blocks[None, :] != loc[:, None], coarse, float("-inf"))
        if KEEP_FIRST:
            has_first = has_first | (loc == 0)


@triton.jit
def _prefill(Q, K, V, Route, Out, N: tl.constexpr, T: tl.constexpr, H: tl.constexpr, KV: tl.constexpr,
             QB: tl.constexpr, QH: tl.constexpr, QT: tl.constexpr,
             KB: tl.constexpr, KH: tl.constexpr, KT: tl.constexpr,
             VB: tl.constexpr, VH: tl.constexpr, VT: tl.constexpr,
             B: tl.constexpr, TS: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
             SKIP_AT: tl.constexpr, SKIP_TILES: tl.constexpr):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if SKIP_TILES:
        tile += tl.where(tile >= SKIP_AT, SKIP_TILES, 0)
    qrow = tile * TS + tl.arange(0, TS)
    dims = tl.arange(0, D)
    kgroup = head // (H // KV)
    q = tl.load(Q + batch * QB + head * QH + qrow[:, None] * QT + dims[None, :], qrow[:, None] < N, 0)
    top = tl.full((TS,), float("-inf"), tl.float32)
    mass = tl.full((TS,), 0, tl.float32)
    total = tl.full((TS, D), 0, tl.float32)
    for i in range(tl.minimum((tile * TS) // B + 1, 8)):
        start = tl.load(Route + ((batch * H + head) * T + tile) * 8 + i) * B
        tokens = start + tl.arange(0, B)
        k = tl.load(K + batch * KB + kgroup * KH + tokens[None, :] * KT + dims[:, None], tokens[None, :] < N, 0)
        value = tl.load(V + batch * VB + kgroup * VH + tokens[:, None] * VT + dims[None, :], tokens[:, None] < N, 0)
        score = tl.dot(q, k) * SCALE
        allowed = (tokens[None, :] <= qrow[:, None]) & (tokens[None, :] < N)
        score = tl.where(allowed, score, float("-inf"))
        new_top = tl.maximum(top, tl.max(score, axis=1))
        scale = tl.exp2(top - new_top)
        p = tl.exp2(score - new_top[:, None])
        mass = mass * scale + tl.sum(p, axis=1)
        total = tl.dot(p.to(tl.bfloat16), value, total * scale[:, None])
        top = new_top
    out = total / mass[:, None]
    tl.store(Out + ((batch * N + qrow[:, None]) * H + head) * D + dims[None, :], out, qrow[:, None] < N)


def block_routes(q, k, *, keep_first=False):
    """Find the fixed-budget shortlist for every query tile and head."""
    assert q.is_cuda and q.dtype == k.dtype == torch.bfloat16
    assert q.size(0) == k.size(0) and q.shape[-2:] == k.shape[-2:]
    batch, heads, tokens, dim = q.shape
    kv = k.size(1)
    assert dim == HEAD_DIM and heads % kv == 0 and q.stride(-1) == k.stride(-1) == 1
    nmeans = triton.cdiv(tokens, GROUP)
    ntiles = triton.cdiv(tokens, QUERY_TILE)
    means = torch.empty((batch, kv, nmeans, dim), device=q.device, dtype=q.dtype)
    route = torch.empty((batch, heads, ntiles, BUDGET), device=q.device, dtype=torch.int32)
    _mean_keys[(nmeans, kv, batch)](k, means, tokens, nmeans, kv, *k.stride()[:3], GROUP, dim, num_warps=4)
    # Split the proxy dot along the head dimension at 8K to fit on-chip memory.
    dot_dim, stages = (32, 1) if nmeans > 256 else (dim, 3)
    _routes[(triton.cdiv(ntiles, 16), heads, batch)](
        q, means, route, tokens, nmeans, ntiles, heads, kv, *q.stride()[:3], 16, BLOCK,
        BLOCK // GROUP, QUERY_TILE, max(16, triton.next_power_of_2(nmeans)), dim, dot_dim, keep_first,
        num_warps=4, num_stages=stages,
    )
    return route


def block_attention(q, k, v, scaling=None, *, record_routes=False, keep_first=False, flash_queries=None):
    """Construct proxies, shortlist blocks, and softmax over the retained token keys."""
    assert k.shape == v.shape and v.dtype == k.dtype and v.stride(-1) == 1
    route = block_routes(q, k, keep_first=keep_first)
    batch, heads, tokens, dim = q.shape
    kv, ntiles = k.size(1), route.size(-2)
    scale = 1 / math.sqrt(dim) if scaling is None else scaling
    out = torch.empty((batch, tokens, heads, dim), device=q.device, dtype=q.dtype)
    skip_at, skip_tiles = ntiles, 0
    if flash_queries is not None:
        start, stop = flash_queries
        assert 0 <= start < stop <= tokens and (stop-start) < tokens and start % QUERY_TILE == stop % QUERY_TILE == 0
        skip_at, skip_tiles = start // QUERY_TILE, (stop-start) // QUERY_TILE
    _prefill[(ntiles-skip_tiles, heads, batch)](
        q, k, v, route, out, tokens, ntiles, heads, kv, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        BLOCK, QUERY_TILE, dim, scale * math.log2(math.e), skip_at, skip_tiles, num_warps=4,
    )
    if flash_queries is not None:
        from torch.nn.attention.bias import causal_lower_right

        rows = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, start:stop], k[:, :, :stop], v[:, :, :stop],
            attn_mask=causal_lower_right(stop-start, stop), scale=scale, enable_gqa=heads != kv,
        )
        out[:, start:stop].copy_(rows.transpose(1, 2))
    return (out, route) if record_routes else out
