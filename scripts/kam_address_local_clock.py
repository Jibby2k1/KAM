"""Derive within-address temporal rank from the fields available at inference."""

from math import log1p

import torch


def order_positions(values, order, *, descending=False):
    ranking = values.gather(1, order).argsort(dim=1, descending=descending, stable=True)
    return order.gather(1, ranking)


def fields(tokens):
    workspace = tokens[..., :8].argmax(-1)
    subject = tokens[..., 40:104].argmax(-1)
    address = workspace * 64 + subject
    clocks = tokens[..., -2:]
    available = tokens[..., 233].bool() & clocks.le(0).all(-1)
    return address, clocks, available


def local_rank(tokens):
    address, clocks, available = fields(tokens)
    ordinal = torch.arange(tokens.shape[1], device=tokens.device)[None].expand(tokens.shape[0], -1)
    order = order_positions(clocks[..., 1], ordinal, descending=True)
    order = order_positions(clocks[..., 0], order, descending=True)
    order = order_positions(available.long(), order, descending=True)
    order = order_positions(address, order)
    by_key = address.gather(1, order)
    cut = torch.cat((torch.ones_like(by_key[:, :1], dtype=torch.bool), by_key[:, 1:].ne(by_key[:, :-1])), dim=1)
    blocks = torch.where(cut, ordinal, 0).cummax(dim=1).values
    by_time = clocks.gather(1, order[..., None].expand_as(clocks))
    breaks = torch.cat((torch.ones_like(cut[:, :1]), cut[:, 1:] | by_time[:, 1:].ne(by_time[:, :-1]).any(-1)), dim=1)
    # Equal date pairs share the earliest ordinal for their group.
    ties = torch.where(breaks, ordinal, 0).cummax(dim=1).values
    return torch.empty_like(order).scatter_(1, order, ties - blocks), available


def shuffled_ranks(ranks, available):
    order = (~available).long().argsort(dim=1, stable=True)
    positions = torch.arange(ranks.shape[1], device=ranks.device)[None]
    count = available.sum(1, keepdim=True)
    offset = (positions + 1).remainder(count.clamp_min(1))
    changed = torch.where(positions < count, order.gather(1, offset.expand_as(order)), order)
    return torch.empty_like(ranks).scatter_(1, order, ranks.gather(1, changed))


def apply_local_clock(raw, recoded, *, sham=False):
    rank, admitted = local_rank(raw)
    if sham:
        rank = shuffled_ranks(rank, admitted)
    clock = -rank.float().log1p() / log1p(24)
    recoded[..., -1] = torch.where(admitted, clock, recoded[..., -1])
    return recoded
