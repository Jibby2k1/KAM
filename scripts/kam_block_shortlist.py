"""Causal, approximate top-four read with an in-request block shortlist."""
import torch
import triton
import triton.language as tl

BLOCK = 64
PROXIES = 4
BUDGETS = (4, 8, 16)


@triton.jit
def _summaries(K, S, NK: tl.constexpr, NS: tl.constexpr, SPAN: tl.constexpr):
    group, head = tl.program_id(0), tl.program_id(1)
    rows = group * SPAN + tl.arange(0, SPAN)
    dim = tl.arange(0, 16)
    key = tl.load(K + (head * NK + rows[:, None]) * 16 + dim[None, :], rows[:, None] < NK, 0)
    count = tl.minimum(NK - group * SPAN, SPAN)
    tl.store(S + (head * NS + group) * 16 + dim, tl.sum(key.to(tl.float32), axis=0) / count)


@triton.jit
def _shortlist(Q, S, Route, NQ: tl.constexpr, NK: tl.constexpr, NS: tl.constexpr, NR: tl.constexpr,
               B: tl.constexpr, P: tl.constexpr, QB: tl.constexpr, SB: tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    rows = tile * QB + tl.arange(0, QB)
    dim = tl.arange(0, 16)
    proxy = tl.arange(0, SB)
    blocks = tl.arange(0, SB // P)
    q = tl.load(Q + (head * NQ + rows[:, None]) * 16 + dim[None, :], rows[:, None] < NQ, 0)
    s = tl.load(S + (head * NS + proxy[None, :]) * 16 + dim[:, None], proxy[None, :] < NS, 0)
    sim = tl.max(tl.reshape(tl.dot(q, s), (QB, SB // P, P)), axis=2)
    current = (rows + NK - NQ) // B
    # Every proxy considered here belongs to a fully visible older block.
    sim = tl.where(blocks[None, :] < current[:, None], sim, float("-inf"))
    base = (head * NQ + rows) * NR
    tl.store(Route + base, current, rows < NQ)
    for slot in tl.static_range(1, NR):
        picked = tl.argmax(sim, axis=1, tie_break_left=True)
        live = tl.max(sim, axis=1) != float("-inf")
        tl.store(Route + base + slot, tl.where(live, picked, -1), rows < NQ)
        sim = tl.where(blocks[None, :] != picked[:, None], sim, float("-inf"))


@triton.jit
def _read(Q, K, V, Route, Out, Ids, NQ: tl.constexpr, NK: tl.constexpr, NR: tl.constexpr,
          B: tl.constexpr, RECORD: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    stop = row + NK - NQ
    candidates = tl.arange(0, NR * B)
    dim = tl.arange(0, 16)
    block = tl.load(Route + (head * NQ + row) * NR + candidates // B)
    key = block * B + candidates % B
    valid = (block >= 0) & (key <= stop)
    q = tl.load(Q + (head * NQ + row) * 16 + dim).to(tl.float32)
    k = tl.load(K + (head * NK + key[:, None]) * 16 + dim[None, :], valid[:, None], 0).to(tl.float32)
    score = tl.where(valid, tl.sum(k * q[None, :], axis=1) * .25, float("-inf"))
    peak = tl.max(score, axis=0)
    value = tl.full((16,), 0, tl.float32)
    total = tl.full((), 0, tl.float32)
    for rank in tl.static_range(4):
        best = tl.max(score, axis=0)
        index = tl.min(tl.where(score == best, key, NK), axis=0)
        live = best != float("-inf")
        weight = tl.exp(best - peak)
        v = tl.load(V + (head * NK + index) * 16 + dim, live, 0).to(tl.float32)
        value += weight * v
        total += weight
        score = tl.where(key != index, score, float("-inf"))
        if RECORD:
            tl.store(Ids + (head * NQ + row) * 4 + rank, tl.where(live, index, -1))
    tl.store(Out + (head * NQ + row) * 16 + dim, value / total)


def block_shortlist_read(q, k, v, budget, *, record=False):
    """Construct four means per block, search proxies, score candidates, gather values."""
    assert q.is_cuda and q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert q.ndim == k.ndim == v.ndim == 4 and k.shape == v.shape and k.shape[:2] == q.shape[:2]
    assert q.size(-1) == k.size(-1) == 16 and 0 < q.size(-2) <= k.size(-2)
    assert budget in BUDGETS
    q, k, v = (t.contiguous() for t in (q, k, v))
    batch, heads, nq, _ = q.shape
    nk = k.size(-2)
    ns = triton.cdiv(nk, BLOCK // PROXIES)
    sb = max(16, triton.next_power_of_2(ns))
    nhead = batch * heads
    summaries = torch.empty((nhead, ns, 16), device=q.device, dtype=q.dtype)
    routes = torch.empty((nhead, nq, budget), device=q.device, dtype=torch.int32)
    out = torch.empty_like(q)
    ids = torch.empty((*q.shape[:-1], 4), device=q.device, dtype=torch.int32) if record else out
    _summaries[(ns, nhead)](k, summaries, nk, ns, BLOCK // PROXIES, num_warps=4)
    _shortlist[(triton.cdiv(nq, 16), nhead)](q, summaries, routes, nq, nk, ns, budget,
                                            BLOCK, PROXIES, 16, sb, num_warps=4)
    _read[(nq, nhead)](q, k, v, routes, out, ids, nq, nk, budget, BLOCK, record,
                        num_warps=4 if budget < 16 else 8)
    return (out, ids) if record else out
