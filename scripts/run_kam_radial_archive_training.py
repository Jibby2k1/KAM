#!/usr/bin/env python3
"""Train matched radial attention kernels and test real revision sequences."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from benchmark_kam_radial_backward import objective, reset
from benchmark_kam_radial_precision import variants
from benchmark_kam_radial_sdpa import labels, predict
from kam_archive_projection import generated_archive_scene, published_questions, scene as archive_scene
from run_kam_bitemporal_decoder import encode, fingerprint, make_decoder, pack, record_order
from run_kam_bitemporal_noise import generate

MODES = ("eager_fp32", "eager_bf16", "augmented_flash_bf16", "balanced_split_flash_bf16")
SCHEDULE = ((16, 159), (48, 33))
LONG_SCHEDULE = ((16, 125), (48, 47), (128, 16), (512, 4))
ARM = "radial_without_position"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def banks_and_order(seed, count, steps, schedule):
    banks = []
    for i, (depth, batch) in enumerate(schedule):
        rows = min(count, 2**20 // depth) if depth > 48 else count
        if depth <= 48:
            scene = generated_archive_scene(seed + 1500 + i, rows, depth, "cpu")
        else:
            scene = generate(seed + 1500 + i, rows, depth, "mixed", "cpu")
        order = record_order(rows, depth, seed + 1773 + i, "cpu")
        rng = torch.Generator().manual_seed(seed + 87137 + i)
        batches = torch.randint(rows, (steps // len(schedule), batch), generator=rng)
        banks.append((tuple(z.to("cuda") for z in scene), order.to("cuda"), batches.to("cuda"),
                      fingerprint(scene, order), sha256(batches.numpy().tobytes()).hexdigest()))
    return banks


def fit(net, mode, banks, steps):
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.002, fused=True)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    losses, tokens = [], 0
    for step in range(1, steps + 1):
        phase = step % len(banks)
        scene, order, batches, _, _ = banks[phase]
        index = batches[(step - 1) // len(banks)]
        x, entity, value = encode(pack(scene, index), order[index], hide_query_clock=True)
        optimizer.param_groups[0]["lr"] = 0.002 * min(step / 100, 1.0) * (
            0.2 + 0.8 * (1 + math.cos(math.pi * step / steps)) / 2)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(predict(net, mode, x), entity, value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        tokens += x.shape[0] * x.shape[1]
        if step in (100, 400, 800, 1600, steps):
            assert torch.isfinite(loss)
            losses.append({"step": step, "tokens": tokens, "loss": float(loss.detach())})
    torch.cuda.synchronize()
    return {"steps": steps, "tokens": tokens, "elapsed_seconds": perf_counter() - start,
            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated() - resident, "sampled_losses": losses}


def archive_panels(fixture, seed, intervention=None):
    questions = published_questions(fixture, "test")
    panels = []
    for depth, _ in SCHEDULE:
        for recode in (0, 1):
            stamp = seed + 1952767 + 1909 * recode + depth
            scenes, rows = archive_scene(questions, depth, stamp, "cpu")
            changed = None
            if intervention:
                scenes, changed = intervention(scenes, stamp)
            order = record_order(len(rows), depth, stamp + 10249, "cpu")
            data = encode(scenes, order, hide_query_clock=True)
            panels.append({"depth": depth, "recode": recode, "data": tuple(z.to("cuda") for z in data),
                           "fixture_sha256": fingerprint(scenes, order), "rows": rows} | (
                           {"changed": changed} if changed is not None else {}))
    return panels


@torch.inference_mode()
def assess(net, mode, panels):
    net.eval()
    result = []
    for panel in panels:
        x, entity, value = panel["data"]
        preds = labels(torch.cat([predict(net, mode, block) for block in x.split(32)]))
        entity_match = preds[:, 0].eq(entity).tolist()
        chosen = preds[:, 1].tolist()
        rows = []
        for detail, address, code, truth in zip(panel["rows"], entity_match, chosen, value.tolist(), strict=True):
            number = detail["by_code"].get(code)
            rows.append({"id": detail["id"], "period": detail["period"], "series": detail["series"],
                         "joint": address and code == truth, "numeric_exact": number == detail["correct_number"],
                         "unbound": number is None})
        result.append({"depth": panel["depth"], "recode": panel["recode"],
                       "fixture_sha256": panel["fixture_sha256"], "outcomes": rows})
    return result


def generated_panels(seed):
    panels = {}
    for depth in (20, 128, 512):
        count = 512
        episode = generate(seed + 3500571 + depth, count, depth, "mixed", "cuda")
        order = record_order(count, depth, seed + 3313751 + depth, "cuda")
        panels[str(depth)] = (episode, order, fingerprint(episode, order))
    return panels


@torch.inference_mode()
def assess_generated(model, mode, panels):
    model.eval()
    outcomes = {}
    for depth, (episode, order, stamp) in panels.items():
        scores = []
        for i in range(0, len(order), 16):
            sl = slice(i, i + 16)
            x, entity, value = encode(pack(episode, sl), order[sl], hide_query_clock=True)
            pred = labels(predict(model, mode, x))
            scores.extend(pred.eq(torch.stack((entity, value), -1)).all(-1).tolist())
        outcomes[depth] = {"fixture_sha256": stamp, "correct": scores}
    return outcomes


def run(a):
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    schedule = LONG_SCHEDULE if a.curriculum == "mixed-long" else SCHEDULE
    assert a.steps >= 100 and a.steps % len(schedule) == 0 and a.bank >= 159
    fixtures = {k: getattr(a, k) for k in ("earlier", "later")}
    benchmarks = {k: json.loads(path.read_text()) for k, path in fixtures.items()}
    assert not ({g["id"] for g in benchmarks["earlier"]["rows"]["test"]} &
                {g["id"] for g in benchmarks["later"]["rows"]["test"]})
    panels = {k: archive_panels(v, a.seed) for k, v in benchmarks.items()}
    initial = make_decoder(ARM, a.seed, "cpu")
    weights = initial.state_dict()
    stamp = sha256(b"".join(t.contiguous().numpy().tobytes() for _, t in sorted(weights.items()))).hexdigest()
    checkpoint = {"arm": ARM, "seed": a.seed, "weights": weights}
    models = variants(checkpoint)
    models = {k: models[k] for k in MODES}
    common_eval = make_decoder(ARM, a.seed, "cuda").eval()
    banks = banks_and_order(a.seed, a.bank, a.steps, schedule)
    generated = generated_panels(a.seed) if a.curriculum == "mixed-long" else {}
    outputs = {}
    shift = a.order % len(MODES)
    for mode in (*MODES[shift:], *MODES[:shift]):
        net = models[mode]
        reset(net, checkpoint)
        training = fit(net, mode, banks, a.steps)
        common_eval.load_state_dict(net.state_dict())
        outputs[mode] = {"training": training,
                         "common_fp32_eval": {k: assess(common_eval, "eager_fp32", p) for k, p in panels.items()},
                         "native_eval": {k: assess(net, mode, p) for k, p in panels.items()}}
        if generated:
            outputs[mode]["common_fp32_generated"] = assess_generated(common_eval, "eager_fp32", generated)
            outputs[mode]["native_generated"] = assess_generated(net, mode, generated)
        net.zero_grad(set_to_none=True)
        print(mode, training["elapsed_seconds"], flush=True)
    return {"seed": a.seed, "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "order": a.order, "start_sha256": stamp, "bank_count": a.bank, "schedule": schedule,
            "bank_shape_counts": [len(bank[0][0]) for bank in banks],
            "bank_sha256": [b[3] for b in banks], "index_sha256": [b[4] for b in banks],
            "source_sha256": {k: digest(path) for k, path in fixtures.items()}, "modes": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("earlier", "later", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--order", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3200)
    parser.add_argument("--bank", type=int, default=16_384)
    parser.add_argument("--curriculum", choices=("short", "mixed-long"), default="short")
    a = parser.parse_args()
    a.output.write_text(json.dumps(run(a), separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
