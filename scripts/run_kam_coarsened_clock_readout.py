#!/usr/bin/env python3
"""Replay frozen guarded KAM checkpoints with week-bucketed publication metadata."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from audit_kam_revision_clock_resolution import bucket, feasible_latest
from benchmark_kam_radial_sdpa import labels
from kam_address_local_clock import local_rank, unique_value
from run_kam_bitemporal_decoder import make_decoder
from run_kam_clock_release_training import source_panels
from run_kam_typed_clock_guard import GuardDecoder


def candidate_days(records, raw, expected, lookup, depth):
    dates = [lookup[r["id"], r["period"]][1]["cutoff"] for r in records]
    cutoff = torch.tensor(dates, device=raw.device)[:, None]
    from_float = (raw[..., -1] * 720).round().long() + cutoff
    fact = raw[..., 233].bool()
    nearest = raw[..., 40:104].argmax(-1)
    workspace = raw[..., :8].argmax(-1)
    numeric = raw[..., 104:232].argmax(-1)
    target_address = fact & nearest.eq(expected[:, None]) & workspace.eq(workspace[:, -1:])
    visibility = []
    for rec, day, all_known, factual, target, value in zip(records, dates, from_float.tolist(), fact.tolist(),
                                                          target_address.tolist(), numeric.tolist(), strict=True):
        group, query = lookup[rec["id"], rec["period"]]
        candidates = group["scenarios"]["clean" if depth == 16 else "noise"]
        assert sorted(group["candidates"][j]["vintage"] for j in candidates) == sorted(d for d, f in zip(all_known, factual, strict=True) if f)
        sequence = sorted((d, v) for d, v, found in zip(all_known, value, target, strict=True) if found)
        possible = feasible_latest(sequence, day, lambda x: bucket(x, "week"), ordered=False)
        gold = group["candidates"][query["gold"]]["vintage"]
        assert any(v is not None and v[0] == gold for v in possible)
        visibility.append({"week_unique_snapshot": len(possible) == 1, "possibilities": len(possible)})
    return cutoff, from_float, fact, target_address, numeric, visibility


@torch.inference_mode()
def run(a):
    torch.set_float32_matmul_precision("highest")
    state = torch.load(a.weights, map_location="cpu", weights_only=True)
    assert state["seed"] == a.seed
    reference = json.loads(a.reference.read_bytes())
    assert reference["seed"] == a.seed and state["reference_sha256"] == sha256(a.reference.read_bytes()).hexdigest()
    paths, source = source_panels(a)
    earlier = json.loads(a.reference_coarse.read_bytes()) if a.reference_coarse else None
    if earlier:
        assert earlier["seed"] == a.seed and earlier["weights_sha256"] == sha256(a.weights.read_bytes()).hexdigest()
    models = {}
    for g, weights in state["models"].items():
        core = make_decoder(g + "_without_position", a.seed, "cuda").eval()
        core.load_state_dict(weights)
        models[g] = GuardDecoder(core, "hard_clock_and_rank").eval()
    result = {}
    for name, panels in source.items():
        assert name != "dallas"
        fixture = json.loads(paths[name].read_bytes())
        lookup = {(g["id"], q["period"]): (g, q) for g in fixture["rows"]["test"] for q in g["questions"]
                  if q["style"] == "canonical" and q["gold"] is not None}
        result[name] = []
        for idx, panel in enumerate(panels):
            raw, key, value = panel["data"]
            cutoff, publish, fact, relevant, payload, visibility = candidate_days(panel["rows"], raw, key, lookup, panel["depth"])
            start = (publish - 1) // 7 * 7 + 1
            predictions = {}
            restricted = {}
            real_future = fact & publish.gt(cutoff)
            truth = torch.stack((key, value), -1)
            admitted_future = {}
            for mode, chosen in (("day", publish), ("week_start", start), ("week_middle", start+3), ("week_end", start+6)):
                raw_input = raw.clone()
                if mode != "day":
                    raw_input[..., -1] = torch.where(fact, (chosen-cutoff).float()/720, raw[..., -1])
                incorrectly_visible = real_future & relevant & raw_input[..., -2:].le(0).all(-1)
                admitted_future[mode] = incorrectly_visible.any(-1).tolist()
                for g, model in models.items():
                    logits = torch.cat([model(x) for x in raw_input.split(32)])
                    result_choices = labels(logits)
                    answer = result_choices.eq(truth)
                    selected = fact & result_choices[:, 1:2].eq(payload)
                    assert selected.sum(1).le(1).all()
                    rows = torch.stack((answer[:, 0], answer[:, 1], (selected & real_future).any(-1),
                                        (selected & real_future & relevant).any(-1), ~selected.any(-1)), dim=-1).tolist()
                    if mode == "day":
                        old = reference["arms"][g+"/hard_clock_and_rank"]["archives"][name][idx]
                        assert old["fixture_sha256"] == panel["fixture_sha256"]
                        assert [x[0] and x[1] for x in rows] == [x["joint"] for x in old["outcomes"]]
                    arm = g+"/"+mode
                    predictions[arm] = rows
                    if earlier:
                        raw_workspace = raw[..., :8].argmax(-1)
                        exposed = fact & raw_input[..., -2:].le(0).all(-1) & raw_workspace.eq(raw_workspace[:, -1:])
                        matching = exposed & raw[..., 40:104].argmax(-1).eq(result_choices[:, 0:1])
                        all_scores = logits[:, 64:].gather(1, payload).masked_fill(~matching, -torch.inf)
                        selected_index = all_scores.argmax(-1, keepdim=True)
                        ranks, eligible = local_rank(raw_input)
                        assert torch.equal(eligible, fact & raw_input[..., -2:].le(0).all(-1))
                        outputs = {
                            "drop_unavailable_output": result_choices[:, 1].masked_fill(~(selected & matching).any(-1), -1),
                            "mask_value_logits": payload.gather(1, selected_index)[:, 0].masked_fill(~matching.any(-1), -1),
                            "exact_eligible_latest": unique_value(payload, matching & ranks.eq(0)),
                        }
                        restricted[arm] = {}
                        for policy, proposed in outputs.items():
                            returned = fact & proposed[:, None].eq(payload)
                            declined = proposed.lt(0)
                            assert not (returned & ~matching).any()
                            future = (returned & real_future).any(-1)
                            if mode in ("day", "week_end"):
                                assert not future.any()
                            restricted[arm][policy] = torch.stack((~declined, proposed.eq(value) & answer[:, 0], future), -1).tolist()
            observations = {"depth": panel["depth"], "recode": panel["recode"], "fixture_sha256": panel["fixture_sha256"],
                            "rows": [{"id": r["id"], "period": r["period"]} | v
                                     for r, v in zip(panel["rows"], visibility, strict=True)],
                            "future_target_admitted": admitted_future, "predictions": predictions}
            if earlier:
                assert observations == earlier["results"][name][idx]
                observations["restricted"] = restricted
            result[name].append(observations)
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "weights_sha256": sha256(a.weights.read_bytes()).hexdigest(),
            "source_sha256": {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}, "results": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("unemployment", "industry", "later-unemployment", "later-industry", "weights", "reference", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.set_defaults(dallas=None)
    p.add_argument("--reference-coarse", type=Path)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
