"""Derive within-address temporal rank from the fields available at inference."""

from math import log1p

import torch


def order_positions(values, order, *, descending=False):
    ranking = values.gather(1, order).argsort(dim=1, descending=descending, stable=True)
    return order.gather(1, ranking)


def fields(tokens, alias=False):
    workspace = tokens[..., :8].argmax(-1)
    start, size, type_index = (8, 32, 232) if alias else (40, 64, 233)
    subject = tokens[..., start:start + size].argmax(-1)
    address = workspace * size + subject
    clocks = tokens[..., -2:]
    available = tokens[..., type_index].bool() & clocks.le(0).all(-1)
    return address, clocks, available


def rank_from_order(address, clocks, order, ordinal):
    by_key = address.gather(1, order)
    cut = torch.cat((torch.ones_like(by_key[:, :1], dtype=torch.bool), by_key[:, 1:].ne(by_key[:, :-1])), dim=1)
    blocks = torch.where(cut, ordinal, 0).cummax(dim=1).values
    by_time = clocks.gather(1, order[..., None].expand_as(clocks))
    breaks = torch.cat((torch.ones_like(cut[:, :1]), cut[:, 1:] | by_time[:, 1:].ne(by_time[:, :-1]).any(-1)), dim=1)
    # Equal date pairs share the earliest ordinal for their group.
    ties = torch.where(breaks, ordinal, 0).cummax(dim=1).values
    # The sort order is a permutation. Gathering its inverse has the same
    # result as scattering and avoids deterministic CUDA scatter in a graph.
    inverse = order.argsort(dim=1)
    return (ties - blocks).gather(1, inverse)


def local_rank(tokens, alias=False):
    address, clocks, available = fields(tokens, alias)
    ordinal = torch.arange(tokens.shape[1], device=tokens.device)[None].expand(tokens.shape[0], -1)
    order = order_positions(clocks[..., 1], ordinal, descending=True)
    order = order_positions(clocks[..., 0], order, descending=True)
    order = order_positions(available.long(), order, descending=True)
    order = order_positions(address, order)
    return rank_from_order(address, clocks, order, ordinal), available


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


def local_features(raw, recoded, *, alias, sham):
    settings = ((False, 39), (True, 231)) if alias else ((False, 39),)
    for is_alias, slot in settings:
        ranks, admitted = local_rank(raw, is_alias)
        if sham:
            ranks = shuffled_ranks(ranks, admitted)
        # The opposite token type uses this coordinate for its own code. The
        # record type flags disambiguate it without increasing model width.
        feature = (ranks.float() + 1).reciprocal()
        recoded[..., slot] = torch.where(admitted, feature, recoded[..., slot])
    return recoded


def typed_address_winners(raw):
    workspace = raw[:, -1, :8].argmax(1)
    name = raw[:, -1, 8:40].argmax(1)
    alias_ranks, aliases = local_rank(raw, alias=True)
    fact_ranks, facts = local_rank(raw)
    same_workspace = raw[..., :8].argmax(-1).eq(workspace[:, None])
    names = raw[..., 8:40].argmax(-1)
    key_ids = raw[..., 40:104].argmax(-1)
    values = raw[..., 104:232].argmax(-1)
    a_winners = same_workspace & aliases & names.eq(name[:, None]) & alias_ranks.eq(0)
    f_winners = same_workspace & facts & fact_ranks.eq(0)
    return unique_value(key_ids, a_winners), key_ids, values, f_winners


def unique_value(values, mask):
    highest = values.masked_fill(~mask, -1).amax(dim=-1)
    nonempty = highest.ge(0)
    agrees = (~mask | values.eq(highest[:, None])).all(dim=-1)
    return highest.masked_fill(~(nonempty & agrees), -1)
