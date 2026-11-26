"""Map real vintage records to fresh per-episode symbols for KAM transfer tests."""

from random import Random

import torch

from run_kam_bitemporal_noise import exact, generate


def published_questions(fixture, split):
    return [(g, q) for g in fixture["rows"][split] for q in g["questions"]
            if q["gold"] is not None and q["style"] == "canonical"]


def scene(cases, depth, seed, device):
    if depth not in (16, 48):
        raise ValueError("archive fixture provides exactly 16 or 48 facts")
    mode = "clean" if depth == 16 else "noise"
    n = len(cases)
    q, aliases, _, alias_gold, _, _ = generate(seed, n, depth, "mixed", "cpu")
    facts, gold_fact, targets, values = [], [], [], []
    session = Random(seed + 59515)
    for idx, (group, item) in enumerate(cases):
        workspace, _, eq, kq = q[idx].tolist()
        allocated_entity = int(aliases[idx, alias_gold[idx], 2])
        address = group["series"], group["observation"]
        footprint = [group["candidates"][i] for i in group["scenarios"][mode]]
        ids = group["scenarios"][mode].copy()
        ordering = session.sample(list(range(depth)), depth)
        payload = session.sample(list(range(128)), depth)
        other_addresses = sorted({(c["series"], c["observation"]) for c in footprint} - {address})
        other_ids = session.sample([x for x in range(64) if x != allocated_entity], len(other_addresses))
        table = {address: allocated_entity, **dict(zip(other_addresses, other_ids, strict=True))}
        bank = []
        value_by_code = {}
        for j in ordering:
            c = footprint[j]
            key = c["series"], c["observation"]
            namespace = workspace
            # Keep competing observations of the queried series in the same
            # workspace. Other series may share the workspace as well.
            if key[0] != address[0] and session.random() < .5:
                namespace = (workspace + session.randrange(1, 8)) % 8
            bank.append((namespace, table[key], payload[j],
                         eq + c["observation"] - group["observation"],
                         kq + c["vintage"] - item["cutoff"]))
            value_by_code[payload[j]] = c["value"]
        g = ordering.index(ids.index(item["gold"]))
        gold_fact.append(g)
        targets.append(bank[g][2])
        values.append({"id": group["id"], "period": item["period"], "series": group["series"],
                       "correct_number": item["value"], "by_code": value_by_code})
        facts.append(bank)
    tensors = (q, aliases, torch.tensor(facts), alias_gold, torch.tensor(gold_fact), torch.tensor(targets))
    result_a, result_f = exact(*tensors[:3], "both")
    assert torch.equal(result_a, tensors[3]) and torch.equal(result_f, tensors[4])
    return tuple(v.to(device) for v in tensors), values


def generated_archive_scene(seed, n, depth, device, *, shared_effective=True, many_future=True, unique_payloads=True):
    """Generate same-observation cohorts with explicit release gaps and unique payloads."""
    query, aliases, _, alias_gold, _, _ = generate(seed, n, depth, "mixed", "cpu")
    rng = torch.Generator().manual_seed(seed + 645031)

    def draw(low, high, size):
        return torch.randint(low, high, size, generator=rng)

    row = torch.arange(n)
    positions = torch.arange(depth).expand(n, -1)
    group = aliases[row, alias_gold, 2]
    prior = draw(2, 8, (n, 1))
    later = draw(0, 3, (n, 1)) if depth == 16 else draw(10, 27, (n, 1))
    if not many_future:
        # Keep same-key history depth fixed and move releases to the past.
        prior, later = prior + later - (later // 4), later // 4
    same = positions <= prior + later
    older = (positions > 0) & (positions <= prior)
    future = same & (positions > prior)
    entity = torch.where(same, group[:, None], (group[:, None] + draw(1, 64, (n, depth))) % 64)
    workspace = torch.where(same | (draw(0, 2, (n, depth)) == 0), query[:, :1],
                            (query[:, :1] + draw(1, 8, (n, depth))) % 8)
    effective = query[:, 2:3] + torch.where(same, 0, draw(-6, 7, (n, depth)))
    offsets = torch.where(older, -positions, torch.where(future, positions - prior, 0))
    recorded = query[:, 3:4] + torch.where(same, offsets, draw(-9, 3 if depth == 16 else 33, (n, depth)))
    payload = torch.rand(n, 128, generator=rng).argsort(-1)[:, :depth]
    # Separate generators keep unaffected inputs identical across controls.
    extra = torch.Generator().manual_seed(seed + 378875)
    effective_shift = torch.randint(1, 7, (n, depth), generator=extra)
    distractor_code = torch.randint(1, 128, (n, depth - 1), generator=extra)
    if not shared_effective:
        effective -= (older | future) * effective_shift
    if not unique_payloads:
        payload[:, 1:] = (payload[:, :1] + distractor_code) % 128
    facts = torch.stack((workspace, entity, payload, effective, recorded), -1)
    permutation = torch.rand(n, depth, generator=rng).argsort(-1)
    facts = facts[row[:, None], permutation]
    fact_gold = permutation.eq(0).long().argmax(-1)
    result_a, result_f = exact(query, aliases, facts, "both")
    assert torch.equal(result_a, alias_gold) and torch.equal(result_f, fact_gold)
    assert torch.all((facts[..., 2] == payload[:, :1]).sum(1) == 1)
    output = (query, aliases, facts, alias_gold, fact_gold, payload[:, 0])
    return tuple(t.to(device) for t in output)
