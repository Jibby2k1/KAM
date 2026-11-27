"""Compute conditioned dot products directly in resident posting lists."""

import torch
import triton
import triton.language as tl

ROWS = 32


@triton.jit
def posting_scores(bank, ids, offsets, selected, query, output, exclude,
                   dim: tl.constexpr, stride: tl.constexpr, rows: tl.constexpr, cols: tl.constexpr,
                   probes: tl.constexpr, batched: tl.constexpr = False):
    list_position = tl.program_id(0)
    tile = tl.program_id(1)
    if batched:
        batch = tl.program_id(2)
        selected += batch * probes
        output += batch * probes * stride
        query += batch * dim
        exclude = tl.load(exclude + batch)
    key = tl.load(selected + list_position)
    start = tl.load(offsets + key)
    stop = tl.load(offsets + key + 1)
    slot_start = tile * rows
    positions = slot_start + tl.arange(0, rows)
    live = start + positions < stop
    if start + slot_start < stop:
        coordinate = tl.arange(0, cols)
        values = tl.load(bank + (start + positions[:, None]) * dim + coordinate[None, :],
                         mask=live[:, None] & (coordinate[None, :] < dim), other=0).to(tl.float32)
        q = tl.load(query + coordinate, mask=coordinate < dim, other=0).to(tl.float32)
        dot = tl.sum(values * q[None, :], axis=1)
        labels = tl.load(ids + start + positions, mask=live, other=-1)
        dot = tl.where(live & (labels != exclude), dot, -float("inf"))
    else:
        dot = tl.full((rows,), -float("inf"), tl.float32)
    tl.store(output + list_position * stride + positions, dot)


def shortlist(values, row_ids, offsets, selected, q, blocked, count, max_list):
    dim = values.shape[1]
    assert values.is_contiguous() and row_ids.is_contiguous() and values.dtype == q.dtype == torch.float16
    stride = triton.cdiv(max_list, ROWS) * ROWS
    scores = torch.empty((len(selected), stride), dtype=torch.float32, device=values.device)
    posting_scores[(len(selected), stride // ROWS)](
        values, row_ids, offsets, selected, q, scores, blocked, dim, stride, ROWS, triton.next_power_of_2(dim),
        len(selected), num_warps=4,
    )
    top = scores.flatten().topk(count).indices
    locations = selected[top // stride]
    return row_ids[offsets[locations] + top % stride]



def batch_shortlist(values, row_ids, offsets, selected, queries, blocked, count, max_list):
    dim = values.shape[1]
    batch, probes = selected.shape
    assert selected.is_contiguous() and blocked.shape == (batch,) and blocked.is_contiguous()
    assert values.is_contiguous() and row_ids.is_contiguous() and queries.is_contiguous()
    assert values.dtype == queries.dtype == torch.float16 and queries.shape == (batch, dim)
    stride = triton.cdiv(max_list, ROWS) * ROWS
    scores = torch.empty((batch, probes * stride), dtype=torch.float32, device=values.device)
    posting_scores[(probes, stride // ROWS, batch)](
        values, row_ids, offsets, selected, queries, scores, blocked, dim, stride, ROWS,
        triton.next_power_of_2(dim), probes, batched=True, num_warps=4,
    )
    top = scores.topk(count, dim=-1).indices
    locations = selected.gather(1, top // stride)
    return row_ids[offsets[locations] + top % stride]
