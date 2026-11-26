#!/usr/bin/env python3
"""Train original KAM sequence blocks to read noisy bitemporal record histories."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch.nn.functional import cross_entropy

from kam.model import KAMSequenceModel
from kam_archive_projection import generated_archive_scene, published_questions, scene as archive_scene
from run_kam_bitemporal_noise import exact, generate

ARMS = {
    "dot_context": ("dot", False),
    "cosine_context": ("cosine", False),
    "cosine_fixed4": ("rbf", False),
    "cosine_learned": ("rbf", False),
    "radial_context": ("rbf", False),
    "dot_hybrid": ("dot", True),
    "radial_hybrid": ("rbf", True),
    "dot_without_absolute": ("dot", False),
    "radial_without_absolute": ("rbf", False),
    "dot_without_position": ("dot", False),
    "radial_without_position": ("rbf", False),
}
WIDTH = 8 + 32 + 64 + 128 + 3 + 2


class TemporalSequence(KAMSequenceModel):
    no_positions = False
    pin_query = None

    def _position_encoding(self, length, device, dtype):
        if self.no_positions:
            return torch.zeros((1, 1, self.d_model), device=device, dtype=dtype)
        positions = super()._position_encoding(length, device, dtype)
        if self.pin_query is not None and length > self.pin_query:
            positions[:, -1] = positions[:, self.pin_query]
        return positions


def encode(scene, order, *, swap=False, hide_query_clock=False):
    query, aliases, facts, gold_a, gold_f, gold_v = scene
    batch, depth = aliases.shape[:2]
    device = query.device
    assert order.shape == (batch, depth * 2)
    clean = torch.zeros(batch, 2 * depth + 1, WIDTH, device=device)
    record_workspace = torch.cat((aliases[..., 0], facts[..., 0], query[:, :1]), 1)
    clean[..., :8].scatter_(2, record_workspace.unsqueeze(-1), 1)
    alias_pos = torch.cat((aliases[..., 1], query[:, 1:2]), 1).unsqueeze(-1)
    clean[:, :depth, 8:40].scatter_(2, alias_pos[:, :depth], 1)
    clean[:, -1:, 8:40].scatter_(2, alias_pos[:, -1:], 1)
    ids = torch.cat((aliases[..., 2], facts[..., 1]), 1)
    clean[:, :-1, 40:104].scatter_(2, ids.unsqueeze(-1), 1)
    values = facts[..., 2].clone()
    if swap:
        values.scatter_(1, gold_f[:, None], ((gold_v + 71) % 128)[:, None])
    clean[:, depth:2*depth, 104:232].scatter_(2, values.unsqueeze(-1), 1)
    clean[:, :depth, 232] = 1
    clean[:, depth:2*depth, 233] = 1
    clean[:, -1, 234] = 1
    absolute = torch.cat((aliases[..., 3:5], facts[..., 3:5]), 1)
    clean[:, :-1, 235:] = (absolute - query[:, None, 2:4]) / 24
    if not hide_query_clock:
        clean[:, -1, 235:] = query[:, 2:4] / 96
    entity = aliases[..., 2].gather(1, gold_a[:, None]).squeeze(-1)
    ordered = clean[:, :-1].gather(1, order.unsqueeze(-1).expand(-1, -1, WIDTH))
    return torch.cat((ordered, clean[:, -1:]), 1), entity, ((gold_v + 71) % 128) if swap else gold_v


def record_order(batch, depth, seed, device):
    rng = torch.Generator().manual_seed(seed)
    return torch.rand(batch, depth * 2, generator=rng).argsort(-1).to(device)


def pack(scene, index):
    return tuple(t[index] for t in scene)


def fingerprint(scene, order):
    return sha256(b"".join(t.cpu().contiguous().numpy().tobytes() for t in (*scene, order))).hexdigest()


@torch.inference_mode()
def score(model, cfg, device):
    model.eval()
    tasks = ((12, "clean"), (20, "mixed"), (32, "irrelevant"), (32, "homonym"), (32, "future_effective"),
             (32, "late_arrival"), (32, "stale"), (32, "mixed"), (64, "irrelevant"), (64, "mixed"),
             (128, "irrelevant"), (128, "mixed"))
    if cfg.extended:
        tasks += ((256, "mixed"), (512, "mixed"))
    results = []
    for depth, profile in tasks:
        scene = generate(cfg.seed + 1_724_359, cfg.eval_examples, depth, profile, device)
        correct_entity, correct_value, joint, follows, switched_entity, pinned_joint = 0, 0, 0, 0, 0, 0
        order = record_order(cfg.eval_examples, depth, cfg.seed + 15308, device)
        robustness = {"alternate_permutation_joint": 0, "alternate_prediction_same": 0,
                      "arrival_order_joint": 0, "clock_shift_joint": 0}
        robust_case = cfg.extended and profile == "mixed" and depth in (20, 64, 128, 256, 512)
        if robust_case:
            alternate = record_order(cfg.eval_examples, depth, cfg.seed + 86770, device)
            arrivals = torch.cat((scene[1][..., 4], scene[2][..., 4]), 1)
            observed = order.gather(1, arrivals.gather(1, order).argsort(dim=-1, stable=True))
            shifted = tuple(t.clone() for t in scene[:3]) + scene[3:]
            shifted[0][:, 2:4] += 120
            shifted[1][..., 3:5] += 120
            shifted[2][..., 3:5] += 120
        batch = min(cfg.eval_batch, max(4, cfg.eval_batch * 64 // depth))
        for start in range(0, cfg.eval_examples, batch):
            sl = slice(start, start + batch)
            x, entity, value = encode(pack(scene, sl), order[sl], hide_query_clock=cfg.no_query_clock)
            changed, unchanged_entity, changed_value = encode(
                pack(scene, sl), order[sl], swap=True, hide_query_clock=cfg.no_query_clock)
            assert torch.equal(entity, unchanged_entity)
            base, counterfactual = model(x), model(changed)
            if depth > 20 and not model.no_positions:
                model.pin_query = 40
                at_fit_position = model(x)
                model.pin_query = None
                pinned_joint += int((at_fit_position[:, :64].argmax(1).eq(entity) &
                                     at_fit_position[:, 64:].argmax(1).eq(value)).sum())
            pred_entity = base[:, :64].argmax(1)
            pred_value = base[:, 64:].argmax(1)
            changed_entity = counterfactual[:, :64].argmax(1)
            pred_changed = counterfactual[:, 64:].argmax(1)
            correct_entity += int(pred_entity.eq(entity).sum())
            correct_value += int(pred_value.eq(value).sum())
            joint += int((pred_entity.eq(entity) & pred_value.eq(value)).sum())
            follows += int((pred_value.eq(value) & pred_changed.eq(changed_value)).sum())
            switched_entity += int((changed_entity != pred_entity).sum())
            if robust_case:
                for key, episode, order_ in (("alternate_permutation_joint", scene, alternate),
                                             ("arrival_order_joint", scene, observed),
                                             ("clock_shift_joint", shifted, order)):
                    intervened, e, v = encode(pack(episode, sl), order_[sl], hide_query_clock=cfg.no_query_clock)
                    assert torch.equal(e, entity) and torch.equal(v, value)
                    seen = model(intervened)
                    pe, pv = seen[:, :64].argmax(1), seen[:, 64:].argmax(1)
                    robustness[key] += int((pe.eq(e) & pv.eq(v)).sum())
                    if key == "alternate_permutation_joint":
                        robustness["alternate_prediction_same"] += int((pe.eq(pred_entity) & pv.eq(pred_value)).sum())
                    if key == "clock_shift_joint" and cfg.no_query_clock:
                        assert torch.equal(x, intervened)
        results.append({"depth_per_bank": depth, "profile": profile, "count": cfg.eval_examples,
                        "correct_entity": correct_entity, "correct_value": correct_value, "joint": joint,
                        "value_follows_intervention": follows, "alias_prediction_changed_on_value_swap": switched_entity,
                        "query_position_control_joint": pinned_joint if depth > 20 and not model.no_positions else None,
                        "robustness": robustness if robust_case else None,
                        "inputs_sha256": fingerprint(scene, order)})
    return results


@torch.inference_mode()
def validation(model, cfg, device):
    model.eval()
    n = min(cfg.eval_examples, 512)
    scene = generate(cfg.seed + 398858, n, 20, "mixed", device)
    order = record_order(n, 20, cfg.seed + 144367, device)
    x, entities, values = encode(scene, order, hide_query_clock=cfg.no_query_clock)
    scores = torch.cat([model(chunk) for chunk in x.split(cfg.eval_batch)])
    ent = scores[:, :64].argmax(1)
    val = scores[:, 64:].argmax(1)
    return {"alias": float(ent.eq(entities).float().mean()), "payload": float(val.eq(values).float().mean()),
            "joint": float((ent.eq(entities) & val.eq(values)).float().mean())}


@torch.inference_mode()
def archive_score(model, cfg, fixture, device, publication_scale=1):
    model.eval()
    cases = published_questions(fixture, "test")
    output = []
    for depth in (16, 48):
        for recode in (0, 1):
            seed = cfg.seed + 1852713 + 1909 * recode + depth
            panel, rows = archive_scene(cases, depth, seed, device)
            if publication_scale != 1:
                cutoff = panel[0][:, None, 3]
                panel[2][..., 4] = cutoff + publication_scale * (panel[2][..., 4] - cutoff)
                found_alias, found_fact = exact(*panel[:3], "both")
                assert torch.equal(found_alias, panel[3]) and torch.equal(found_fact, panel[4])
            order = record_order(len(rows), depth, seed + 10249, device)
            gaps = ((0, 0), (8, 0), (0, 8), (8, 8)) if cfg.archive_boundary_diagnostic else ((0, 0),)
            for effective_gap, publication_gap in gaps:
                if effective_gap or publication_gap:
                    candidate = (*panel[:2], panel[2].clone(), *panel[3:])
                    # The queried address fixes its observation. Offset eligible
                    # publications together, preserving the future set and order.
                    candidate[2][..., 3] -= effective_gap
                    eligible = candidate[2][..., 4] <= candidate[0][:, None, 3]
                    candidate[2][..., 4] -= publication_gap * eligible
                    found_alias, found_fact = exact(*candidate[:3], "both")
                    assert torch.equal(found_alias, candidate[3]) and torch.equal(found_fact, candidate[4])
                else:
                    candidate = panel
                for start in range(0, len(rows), cfg.eval_batch):
                    end = start + cfg.eval_batch
                    x, entities, values = encode(pack(candidate, slice(start, end)), order[start:end],
                                                hide_query_clock=cfg.no_query_clock)
                    result = model(x)
                    pred_entity = result[:, :64].argmax(1)
                    pred_value = result[:, 64:].argmax(1)
                    for row, ent, target, pred in zip(rows[start:end], pred_entity.eq(entities).tolist(),
                                                      values.tolist(), pred_value.tolist(), strict=True):
                        number = row["by_code"].get(pred)
                        output.append({"id": row["id"], "period": row["period"], "series": row["series"],
                                       "depth": depth, "recode": recode,
                                       "effective_gap_intervention": effective_gap,
                                       "publication_gap_intervention": publication_gap,
                                       "entity": ent, "record": pred == target, "joint": ent and pred == target,
                                       "numeric_exact": number == row["correct_number"],
                                       "unbound_payload": number is None})
    return output


def matched_archive_masks(cases, scene):
    query, aliases, facts, gold_alias, gold_fact, _ = scene
    correct_address = aliases[..., 2].gather(1, gold_alias[:, None])
    same = ((facts[..., 0] == query[:, :1]) & (facts[..., 1] == correct_address)
            & (facts[..., 3] == query[:, 2:3]))
    queried_releases = torch.tensor(
        [[group["candidates"][q["gold"]]["vintage"] - item["cutoff"]
          for q in group["questions"] if q["style"] == "canonical" and q["gold"] is not None]
         for group, item in cases], device=query.device,
    )
    gap = facts[..., 4] - query[:, 3:4]
    must_keep = same & gap[..., None].eq(queried_releases[:, None, :]).any(-1)
    assert torch.all(must_keep.sum(-1) == 2)
    nearest = gap.abs().masked_fill(~(same & ~must_keep), torch.iinfo(gap.dtype).max).argmin(-1)
    assert torch.all(same.sum(-1) >= 3)
    remove_same = same & ~must_keep
    remove_same.scatter_(1, nearest[:, None], False)
    remove_other = ~same & ((~same).cumsum(-1) <= remove_same.sum(-1, keepdim=True))
    assert torch.all(remove_same.sum(-1) == remove_other.sum(-1))
    assert torch.all((same & ~remove_same).sum(-1) == 3)
    assert must_keep.gather(1, gold_fact[:, None]).all()
    return {"observed": torch.ones_like(same), "three_releases": ~remove_same, "load_control": ~remove_other}, same


@torch.inference_mode()
def matched_archive_score(model, cfg, fixture, device):
    """Exclude masked keys at every context layer and retain their position slots."""
    model.eval()
    cases = published_questions(fixture, "test")
    output = []
    active_tokens = None

    def mask_keys(_module, _args, scores):
        return scores.masked_fill(~active_tokens[:, None, None, :], torch.finfo(scores.dtype).min)

    hooks = [block.context.score.register_forward_hook(mask_keys) for block in model.blocks]
    assert all(block.memory is None for block in model.blocks)
    depth = 48
    try:
        for recode in (0, 1):
            seed = cfg.seed + 1852713 + 1909 * recode + depth
            panel, rows = archive_scene(cases, depth, seed, device)
            masks, same = matched_archive_masks(cases, panel)
            order = record_order(len(rows), depth, seed + 10249, device)
            for clock in (1, 3):
                query, aliases, facts, gold_alias, gold_fact, gold_value = panel
                facts = facts.clone()
                cutoff = query[:, 3:4]
                facts[..., 4] = cutoff + clock * (facts[..., 4] - cutoff)
                candidate = (query, aliases, facts, gold_alias, gold_fact, gold_value)
                for arm, fact_mask in masks.items():
                    # Independent maximum over release dates for the fixed address.
                    eligible = same & fact_mask & (facts[..., 4] <= cutoff)
                    assert eligible.any(-1).all()
                    by_release = facts[..., 4].masked_fill(~eligible, torch.iinfo(facts.dtype).min).argmax(-1)
                    assert torch.equal(by_release, gold_fact)
                    aliases_visible = torch.ones_like(fact_mask)
                    keep = torch.cat((aliases_visible, fact_mask), 1).gather(1, order)
                    keep = torch.cat((keep, aliases_visible[:, :1]), 1)
                    counts = fact_mask.sum(-1).tolist()
                    for start in range(0, len(rows), cfg.eval_batch):
                        end = start + cfg.eval_batch
                        active_tokens = keep[start:end]
                        x, entities, values = encode(pack(candidate, slice(start, end)), order[start:end],
                                                    hide_query_clock=cfg.no_query_clock)
                        logits = model(x)
                        correct = (logits[:, :64].argmax(-1).eq(entities)
                                   & logits[:, 64:].argmax(-1).eq(values)).tolist()
                        for row, count, hit in zip(rows[start:end], counts[start:end], correct, strict=True):
                            output.append([row["id"], row["period"], depth, recode, clock, arm, count, hit])
    finally:
        for hook in hooks:
            hook.remove()
    return output


def include_clock_boundaries(scene, seed, count):
    query, _, records, _, gold_f, _ = scene
    g = torch.Generator().manual_seed(seed)
    for column in range(count):
        adjust = torch.rand(len(query), generator=g).to(query.device) < .5
        delta = (query[:, 2 + column] - records[..., 3 + column].gather(1, gold_f[:, None]).squeeze(1)) * adjust
        records[..., 3 + column] += delta[:, None]
    alias, fact = exact(*scene[:3], "both")
    assert torch.equal(alias, scene[3]) and torch.equal(fact, gold_f)


def make_decoder(arm, seed, device):
    score_kind, has_memory = ARMS[arm]
    architecture = dict(task="regression", d_model=64, num_heads=4, num_layers=2, num_supports=24,
                        max_seq_len=257, input_dim=WIDTH, output_dim=192,
                        expose_memory_weights=False, position_mode="sinusoidal", ffn_expansion=2)
    torch.manual_seed(seed)
    reference = KAMSequenceModel(**architecture, score_type="dot", use_memory=True)
    torch.manual_seed(seed)
    calibration = {}
    if arm in ("cosine_fixed4", "cosine_learned"):
        calibration = {"radial_metric": "isotropic"}
    if arm == "cosine_fixed4":
        calibration |= {"bandwidth": "fixed", "init_bandwidth": 0.5}
    model = TemporalSequence(**architecture, score_type=score_kind, use_memory=has_memory, **calibration)
    model.no_positions = arm.endswith(("_without_absolute", "_without_position"))
    if arm.endswith("_without_position"):
        for block in model.blocks:
            block.context.relative_bias = None
    for name, parameter in model.named_parameters():
        if name.endswith("raw_metric") and arm in ("cosine_fixed4", "cosine_learned"):
            parameter.requires_grad_(False)
    model_keys = model.state_dict().keys()
    common = {k: v for k, v in reference.state_dict().items() if k in model_keys}
    model.load_state_dict(common, strict=False)
    return model.to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--bank", type=int, default=16_384)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--eval-examples", type=int, default=1024)
    p.add_argument("--schedule", choices=("short", "wide"), default="short")
    p.add_argument("--no-query-clock", action="store_true")
    p.add_argument("--extended", action="store_true")
    p.add_argument("--archive-fixture", type=Path)
    p.add_argument("--archive-thin-fixture", type=Path)
    p.add_argument("--archive-scale-check", action="store_true")
    p.add_argument("--archive-mask-check", action="store_true")
    p.add_argument("--training-data", choices=("procedural", "procedural_corrections", "procedural_boundary",
                                                "procedural_dual_boundary", "procedural_archive_topology", "archive"),
                   default="procedural")
    p.add_argument("--archive-boundary-diagnostic", action="store_true")
    p.add_argument("--spread-effective-dates", action="store_true")
    p.add_argument("--mostly-earlier-revisions", action="store_true")
    p.add_argument("--allow-duplicate-distractor-payloads", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    cfg = p.parse_args()
    assert cfg.steps > 0 and cfg.batch <= cfg.bank
    if (cfg.training_data in ("archive", "procedural_archive_topology") or cfg.archive_boundary_diagnostic) and cfg.archive_fixture is None:
        p.error("Archive training and diagnostics require --archive-fixture")
    if (cfg.archive_thin_fixture or cfg.archive_scale_check or cfg.archive_mask_check) and cfg.archive_fixture is None:
        p.error("Archive interventions require --archive-fixture")
    if cfg.archive_mask_check and cfg.arm not in ("dot_without_absolute", "radial_without_absolute"):
        p.error("Matched archive masking requires a context-only relative-position arm")
    archive_generation = {"shared_effective": not cfg.spread_effective_dates,
                          "many_future": not cfg.mostly_earlier_revisions,
                          "unique_payloads": not cfg.allow_duplicate_distractor_payloads}
    if not all(archive_generation.values()) and cfg.training_data != "procedural_archive_topology":
        p.error("Archive generation controls require --training-data procedural_archive_topology")
    fixture = json.loads(cfg.archive_fixture.read_text()) if cfg.archive_fixture else None
    torch.set_num_threads(4 if cfg.device == "cuda" else 1)
    torch.set_float32_matmul_precision("highest")
    score_kind, _ = ARMS[cfg.arm]
    device = torch.device(cfg.device)
    model = make_decoder(cfg.arm, cfg.seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, fused=device.type == "cuda")
    rng = torch.Generator(device=device).manual_seed(cfg.seed + 87137)
    banks = []
    training_digests = []
    schedule = ((12, "clean", cfg.batch), (20, "mixed", cfg.batch))
    if cfg.schedule == "wide":
        assert cfg.batch == 128
        schedule = ((12, "clean", 166), (20, "mixed", 104), (32, "mixed", 65), (64, "mixed", 33))
    if cfg.archive_fixture:
        # Identical depths, batches, and token exposure across training sources.
        schedule = ((16, "mixed", 159), (48, "mixed", 33))
    archive_cases = published_questions(fixture, "train") if cfg.training_data == "archive" else None
    for phase, (d, noise, batch) in enumerate(schedule):
        assert batch <= cfg.bank
        phase_seed = cfg.seed + 1500 + phase
        if archive_cases is not None:
            sampled = Random(phase_seed).choices(archive_cases, k=cfg.bank)
            scene, _ = archive_scene(sampled, d, phase_seed, device)
        elif cfg.training_data == "procedural_archive_topology":
            scene = generated_archive_scene(phase_seed, cfg.bank, d, device, **archive_generation)
        else:
            if cfg.training_data != "procedural" and noise == "mixed":
                noise = "correction_mix"
            scene = generate(phase_seed, cfg.bank, d, noise, device)
            if cfg.training_data in ("procedural_boundary", "procedural_dual_boundary"):
                include_clock_boundaries(scene, phase_seed + 15301, 1 + (cfg.training_data == "procedural_dual_boundary"))
        order = record_order(cfg.bank, d, cfg.seed + 1773 + phase, device)
        training_digests.append(fingerprint(scene, order))
        banks.append((scene, order, batch))

    start = perf_counter()
    traces, tokens = [], 0
    intervals = {cfg.steps, *[step for step in (100, 400, 800, 1600, 2500, 3200) if step <= cfg.steps]}
    for step in range(1, cfg.steps + 1):
        model.train()
        scene, order, batch = banks[step % len(banks)]
        subset = torch.randint(cfg.bank, (batch,), device=device, generator=rng)
        x, ent, val = encode(pack(scene, subset), order[subset], hide_query_clock=cfg.no_query_clock)
        opt.param_groups[0]["lr"] = 0.002 * min(step / 100, 1.0) * (0.2 + 0.8 * (1 + math.cos(math.pi * step / cfg.steps)) / 2)
        update = model(x)
        entity_loss = cross_entropy(update[:, :64], ent)
        value_loss = cross_entropy(update[:, 64:], val)
        loss = entity_loss + value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens += x.shape[0] * x.shape[1]
        if step in intervals:
            assert torch.isfinite(loss)
            val_result = validation(model, cfg, device)
            item = {"step": step, "entity_loss": float(entity_loss.detach()), "value_loss": float(value_loss.detach()),
                    "validation": val_result, "train_tokens": tokens, "elapsed_seconds": perf_counter() - start}
            traces.append(item)
            print(json.dumps(item), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = perf_counter() - start
    output = {"arm": cfg.arm, "seed": cfg.seed, "steps": cfg.steps, "batch": cfg.batch, "bank": cfg.bank,
              "schedule": cfg.schedule, "hide_query_clock": cfg.no_query_clock,
              "training_data": cfg.training_data, "training_depth_batches": [(d, b) for d, _, b in schedule],
              "eval_examples": cfg.eval_examples, "train_tokens": tokens, "train_seconds": train_seconds,
              "train_input_sha256": training_digests, "torch": str(torch.__version__),
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "context_temperatures": [block.context.score.bandwidth().detach().cpu().square().reciprocal().tolist()
                                       for block in model.blocks] if score_kind == "rbf" else None,
              "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "traces": traces,
              "test": score(model, cfg, device), "elapsed_seconds": perf_counter() - start}
    if fixture is not None:
        output["archive_fixture_sha256"] = sha256(cfg.archive_fixture.read_bytes()).hexdigest()
        output["archive_test_years"] = fixture.get("test_years", [2013, 2023])
        output["archive"] = archive_score(model, cfg, fixture, device)
        if cfg.archive_mask_check:
            output["archive_masks"] = matched_archive_score(model, cfg, fixture, device)
        if cfg.archive_scale_check:
            output["archive_scaled"] = archive_score(model, cfg, fixture, device, publication_scale=3)
        if cfg.archive_thin_fixture:
            thin = json.loads(cfg.archive_thin_fixture.read_text())
            output["archive_thin_sha256"] = sha256(cfg.archive_thin_fixture.read_bytes()).hexdigest()
            output["archive_thin"] = archive_score(model, cfg, thin, device)
            if cfg.archive_scale_check:
                output["archive_thin_scaled"] = archive_score(model, cfg, thin, device, publication_scale=3)
    if cfg.training_data == "procedural_archive_topology":
        output["archive_generation"] = archive_generation
    cfg.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if cfg.checkpoint is not None:
        torch.save({"weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "arm": cfg.arm, "seed": cfg.seed, "hide_query_clock": cfg.no_query_clock,
                    "train_input_sha256": training_digests, "train_tokens": tokens}, cfg.checkpoint)


if __name__ == "__main__":
    main()
