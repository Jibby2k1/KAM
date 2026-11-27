#!/usr/bin/env python3
"""Compare dense, readout-only and hard final-layer attention on matched radial decoders."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_radial_precision import stable_forward, variants
from benchmark_kam_radial_sdpa import labels
from run_kam_bitemporal_decoder import encode, fingerprint, make_decoder, record_order
from run_kam_bitemporal_noise import generate
from run_kam_packed_clock_inference import profile
from run_kam_radial_archive_training import archive_panels
from run_kam_radial_curriculum_order import state_sha
from run_kam_temporal_clock_encoding import ClockDecoder, SCHEDULE, banks, exposure_order, fit, recode
from run_kam_temporal_correction_curriculum import intervention

MODES = ("full_fp32", "readout_fp32", "top1", "top4", "top16", "top64", "full_flash", "readout_flash")


def hard_read(attention, inputs, k):
    n, length, _ = inputs.shape
    projected = attention.qkv(inputs).view(n, length, 3, attention.num_heads, attention.head_dim)
    q, keys, values = projected.permute(2, 0, 3, 1, 4).unbind(0)
    scores = attention.score(q[..., -1:, :], keys)
    if k is None:
        update = scores.softmax(-1) @ values
        retained = None
    else:
        score, positions = scores.topk(min(k, length), dim=-1)
        kept = values.gather(2, positions.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, values.shape[-1]))
        update = score.softmax(-1) @ kept
        retained = (score.logsumexp(-1) - scores.logsumexp(-1)).exp().squeeze(-1)
    return attention.output(update.transpose(1, 2).reshape(n, 1, attention.d_model)), retained


def last_output(model, tokens, mode):
    # With one final readout and no later mixing, other rows cannot affect the result of the last block.
    assert not model.use_memory and model.no_positions and model.regression_pool == "last"
    hidden = model.input_layer(tokens)
    for block in model.blocks[:-1]:
        hidden, _, _ = block(hidden)
    tail = model.blocks[-1]
    assert tail.context.window is None and tail.context.relative_bias is None and tail.context.dropout.p == 0
    encoded = tail.context_norm(hidden)
    if mode == "readout_flash":
        mixed, _ = stable_forward(tail.context, encoded, balanced=True, split_bias=True, last_only=True)
        mass = None
    else:
        mixed, mass = hard_read(tail.context, encoded, int(mode[3:]) if mode.startswith("top") else None)
    hidden = hidden[:, -1:, :] + mixed
    hidden = hidden + tail.ffn(tail.ffn_norm(hidden))
    return model.readout(model.final_norm(hidden[:, 0])), mass


def infer(networks, tokens, mode):
    flash = mode.endswith("flash")
    model = networks["balanced_split_flash_bf16" if flash else "eager_fp32"]
    if flash:
        with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return (model(tokens), None) if mode == "full_flash" else last_output(model, tokens, mode)
    return (model(tokens), None) if mode == "full_fp32" else last_output(model, tokens, mode)


@torch.inference_mode()
def assess(networks, tokens, target, batch, seed, *, repeats=0):
    if repeats:
        times = {k: profile(lambda: infer(networks, tokens[:batch], k), repeats)
                 for k in Random(seed).sample(list(MODES), len(MODES))}
    else:
        times = {}
    observed = {}
    scores = {}
    for method in MODES:
        values = [infer(networks, x, method) for x in tokens.split(batch)]
        full = torch.cat([p for p, _ in values]).float()
        assert torch.isfinite(full).all()
        scores[method] = full
        predicted = labels(full)
        flags = predicted.eq(target)
        base = scores["full_flash" if method.endswith("flash") else "full_fp32"]
        output = {"decisions": predicted.tolist(), "correct": flags.tolist(),
                  "largest_logit_difference": float((full - base).abs().max())}
        if method in times:
            output["timing"] = times[method]
        if method.startswith("top"):
            mass = torch.cat([m for _, m in values]).float()
            output["per_head_retained"] = torch.quantile(mass, torch.tensor([.05, .5, .95], device=mass.device), dim=0).tolist()
        observed[method] = output
    return observed


def state(a):
    if a.fit_reference:
        reference = json.loads(a.fit_reference.read_bytes())
        assert a.seed == reference["seed"] and reference["training_domain"] == "mixed" and reference["steps"] == 3200
        original = make_decoder("radial_without_position", a.seed, "cuda")
        cached = banks(a.seed, "mixed", SCHEDULE)
        assert {str(depth): row[-1] for depth, row in cached.items()} == reference["banks"]
        model = ClockDecoder(original, "log_episode_mean")
        row = reference["results"]["radial/log_episode_mean"]
        assert state_sha(original, common=True) == row["initial_sha256"]
        trained = fit(model, cached, exposure_order(SCHEDULE, a.seed, False))
        assert trained["losses"] == row["fit"]["losses"]
        return {"arm": "radial_without_position", "seed": a.seed, "weights": original.state_dict()}, (
            "log_episode_mean", trained, sha256(a.fit_reference.read_bytes()).hexdigest())
    value = a.checkpoint.read_bytes()
    assert sha256(value).hexdigest() == a.checkpoint_sha256
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["arm"] == "radial_without_position" and checkpoint["hide_query_clock"]
    return checkpoint, ("linear", None, a.checkpoint_sha256)


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    checkpoint, (clock, fit_, source) = state(a)
    networks = variants(checkpoint)
    procedural = {}
    for depth, width in ((48, 32), (128, 16), (512, 8), (1024, 4)):
        scene = generate(a.seed + 982433 + depth, 192, depth, "mixed", "cuda")
        permutation = record_order(192, depth, a.seed + 74166, "cuda")
        changed, _ = intervention(scene)
        for name, situation in (("standard", scene), ("corrections", changed)):
            x, entity, value = encode(situation, permutation, hide_query_clock=True)
            metrics = assess(networks, recode(x, clock), torch.stack((entity, value), dim=-1), width, a.seed + depth,
                             repeats=24 if name == "standard" else 0)
            procedural[f"{name}/{depth}"] = {"fixture_sha256": fingerprint(situation, permutation),
                                               "results": metrics, "batch": width}
        print(depth, flush=True)
    archived = {}
    for tag, file in (("dallas", a.dallas), ("eurostat", a.eurostat)):
        archived[tag] = []
        for panel in archive_panels(json.loads(file.read_bytes()), a.seed):
            tokens, entity, value = panel["data"]
            measurements = assess(networks, recode(tokens, clock), torch.stack((entity, value), dim=-1), 32, a.seed)
            rows = [{"id": r["id"], "period": r["period"], "series": r["series"]} for r in panel["rows"]]
            archived[tag].append({"depth": panel["depth"], "recode": panel["recode"],
                                  "fixture_sha256": panel["fixture_sha256"], "rows": rows, "results": measurements})
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "clock": clock, "fit": fit_,
            "weights_sha256": state_sha(networks["eager_fp32"]), "reference_sha256": source,
            "archives_sha256": {k: sha256(getattr(a, k).read_bytes()).hexdigest() for k in ("dallas", "eurostat")},
            "procedural": procedural, "archives": archived}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("dallas", "eurostat", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--seed", type=int, required=True)
    choose = p.add_mutually_exclusive_group(required=True)
    choose.add_argument("--fit-reference", type=Path)
    choose.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-sha256")
    a = p.parse_args()
    assert bool(a.checkpoint) == bool(a.checkpoint_sha256)
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
