#!/usr/bin/env python3
"""Measure learned confidence on answerable and unavailable real revision cutoffs."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import torch

from kam_archive_projection import published_questions, scene as archive_scene
from run_kam_bitemporal_decoder import encode, make_decoder, pack, record_order


@torch.inference_mode()
def score(model, fixture, seed, scales, device, batch):
    cases = [(group, item) for group, item in published_questions(fixture, "test") if item["period"] == "earlier"]
    unavailable = [next(q for q in g["questions"] if q["period"] == "unavailable" and q["style"] == "canonical")
                   for g, _ in cases]
    assert len({g["id"] for g, _ in cases}) == len(cases)
    assert all(item["gold"] is None for item in unavailable)
    shifts = torch.tensor([q["cutoff"] - later["cutoff"] for (_, q), later in zip(cases, unavailable, strict=True)],
                          device=device)[:, None]
    results = []
    for recode in (0, 1):
        recode_seed = seed + 1852713 + 1909 * recode + 48
        scene, _ = archive_scene(cases, 48, recode_seed, device)
        query, aliases, facts, gold_alias, gold_fact, gold_value = scene
        order = record_order(len(cases), 48, recode_seed + 10249, device)
        entity = aliases[..., 2].gather(1, gold_alias[:, None])
        address = ((facts[..., 0] == query[:, :1]) & (facts[..., 1] == entity)
                   & (facts[..., 3] == query[:, 2:3]))
        cutoff = query[:, 3:4]
        for scale in scales:
            for known, shift in ((True, 0), (False, shifts)):
                probe = facts.clone()
                probe[..., 4] = cutoff + scale * (probe[..., 4] + shift - cutoff)
                eligible = address & (probe[..., 4] <= cutoff)
                assert bool(eligible.any(-1).all()) if known else not bool(eligible.any())
                if known:
                    max_release = probe[..., 4].masked_fill(~eligible, torch.iinfo(facts.dtype).min).argmax(-1)
                    assert torch.equal(max_release, gold_fact)
                candidate = (query, aliases, probe, gold_alias, gold_fact, gold_value)
                for start in range(0, len(cases), batch):
                    sl = slice(start, start + batch)
                    x, entity_label, payload_label = encode(pack(candidate, sl), order[sl], hide_query_clock=True)
                    logits = model(x)
                    payload = logits[:, 64:]
                    prediction = payload.argmax(1)
                    value_probability = payload.softmax(1).amax(1)
                    top2 = payload.topk(2, dim=1).values
                    value_margin = top2[:, 0] - top2[:, 1]
                    entity_matches = logits[:, :64].argmax(1).eq(entity_label)
                    valid_selection = (eligible[sl] & facts[sl, :, 2].eq(prediction[:, None])).any(-1)
                    valid_selection &= entity_matches
                    correct_selection = entity_matches & prediction.eq(payload_label) & known
                    assert (correct_selection <= valid_selection).all()
                    predictions = torch.stack((value_probability, value_margin, valid_selection, correct_selection), 1).tolist()
                    for (group, _), (prob, margin, valid, correct) in zip(cases[sl], predictions, strict=True):
                        results.append([group["id"], group["series"], recode, scale, known, prob, margin,
                                        bool(valid), bool(correct)])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("checkpoint", "calibration", "test", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4 if device.type == "cuda" else 2)
    torch.set_float32_matmul_precision("highest")
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    assert state["hide_query_clock"]
    model = make_decoder(state["arm"], state["seed"], device).eval()
    model.load_state_dict(state["weights"])
    calibration = args.calibration.read_bytes()
    test = args.test.read_bytes()
    output = {"arm": state["arm"], "seed": state["seed"], "train_tokens": state["train_tokens"],
              "train_input_sha256": state["train_input_sha256"],
              "calibration_sha256": sha256(calibration).hexdigest(), "test_sha256": sha256(test).hexdigest(),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
              "calibration": score(model, json.loads(calibration), state["seed"], (1,), device, args.batch),
              "test": score(model, json.loads(test), state["seed"], (1, 3), device, args.batch)}
    args.output.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
