"""Tiled exact top-four scoring with sparse value reads on inference-only live KV."""
import torch
import triton
import triton.language as tl


@triton.jit
def _local_candidates(Q, K, Scores, Indices, NQ: tl.constexpr, NK: tl.constexpr, TILES: tl.constexpr,
                      QB: tl.constexpr, KB: tl.constexpr, CAUSAL: tl.constexpr):
    # Each program handles a query tile and one key tile; future tiles are skipped.
    qi, ki, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qr = qi * QB + tl.arange(0, QB)
    kr = ki * KB + tl.arange(0, KB)
    dim = tl.arange(0, 16)
    if not CAUSAL or ki * KB <= (qi + 1) * QB - 1 + NK - NQ:
        q = tl.load(Q + ((head*NQ + qr[:, None])*16 + dim[None, :]), qr[:, None] < NQ, 0)
        k = tl.load(K + ((head*NK + kr[None, :])*16 + dim[:, None]), kr[None, :] < NK, 0)
        score = tl.dot(q, k) * .25
        allowed = kr[None, :] < NK
        if CAUSAL:
            allowed = allowed & (kr[None, :] <= qr[:, None] + NK - NQ)
        score = tl.where(allowed, score, float("-inf"))
        offset = ((head*NQ + qr)*TILES + ki)*4
        for r in tl.static_range(4):
            best = tl.max(score, 1)
            index = tl.argmax(score, 1, tie_break_left=True)
            tl.store(Scores + offset + r, best, qr < NQ)
            tl.store(Indices + offset + r, ki*KB + index, qr < NQ)
            score = tl.where((kr[None, :] - ki*KB) == index[:, None], float("-inf"), score)


@triton.jit
def _sparse_mix(V, Scores, Indices, Out, NQ: tl.constexpr, NK: tl.constexpr, TILES: tl.constexpr,
                KB: tl.constexpr, C: tl.constexpr, CAUSAL: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    candidates = tl.arange(0, C)
    good = candidates < TILES*4
    if CAUSAL:
        good = good & (candidates//4*KB <= row + NK - NQ)
    off = (head*NQ + row)*TILES*4 + candidates
    scores = tl.load(Scores + off, good, float("-inf"))
    dim = tl.arange(0, 16)
    out = tl.full((16,), 0, tl.float32)
    denom = tl.full((), 0, tl.float32)
    peak = tl.max(scores, 0)
    for r in tl.static_range(4):
        best = tl.max(scores, 0)
        position = tl.argmax(scores, 0, tie_break_left=True)
        weight = tl.exp(best - peak)
        index = tl.load(Indices + (head*NQ + row)*TILES*4 + position, best != float("-inf"), 0)
        value = tl.load(V + (head*NK + index)*16 + dim).to(tl.float32)
        out += weight * value
        denom += weight
        scores = tl.where(candidates == position, float("-inf"), scores)
    tl.store(Out + (head*NQ + row)*16 + dim, out / denom)


def exact_sparse_read(q, k, v, *, causal):
    """Score eligible keys in tiles, gather four values, avoid a full score matrix."""
    assert q.is_cuda and q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert k.shape == v.shape and k.shape[:2] == q.shape[:2] and q.size(-1) == k.size(-1) == 16
    batch, heads, queries, _ = q.shape
    keys = k.size(-2)
    assert 0 < queries <= keys
    q, k, v = (t.contiguous() for t in (q, k, v))
    qb, kb = 32, 128
    tiles = triton.cdiv(keys, kb)
    scratch = (batch * heads, queries, tiles, 4)
    scores = torch.empty(scratch, device=q.device, dtype=torch.float32)
    indices = torch.empty(scratch, device=q.device, dtype=torch.int32)
    out = torch.empty_like(q)
    _local_candidates[(triton.cdiv(queries, qb), tiles, batch*heads)](
        q, k, scores, indices, queries, keys, tiles, qb, kb, causal,
    )
    _sparse_mix[(queries, batch*heads)](v, scores, indices, out, queries, keys, tiles, kb,
                                       triton.next_power_of_2(tiles*4), causal, num_warps=4)
    return out
