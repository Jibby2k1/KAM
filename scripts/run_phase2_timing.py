from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from kam.capacity import approximate_flops, parameter_count
from kam.factory import make_model
from kam.utils import choose_device, save_json, set_seed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_one(model: torch.nn.Module, inputs: torch.Tensor, *, device: torch.device, iterations: int, warmup: int, backward: bool, precision: str) -> dict[str, float]:
    model.train(backward)
    autocast_enabled = precision in {"amp", "bf16", "fp16"} and device.type == "cuda"
    dtype = torch.bfloat16 if precision in {"amp", "bf16"} else torch.float16
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            output = model(inputs)
            loss = output.float().square().mean()
        if backward:
            loss.backward()
            model.zero_grad(set_to_none=True)
    _sync(device)
    samples: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(iterations):
        _sync(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            output = model(inputs)
            loss = output.float().square().mean()
        if backward:
            loss.backward()
            model.zero_grad(set_to_none=True)
        _sync(device)
        samples.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    return {
        "median_ms": float(statistics.median(samples) * 1000.0),
        "iqr_ms": float((np.quantile(samples, 0.75) - np.quantile(samples, 0.25)) * 1000.0),
        "p90_ms": float(np.quantile(samples, 0.90) * 1000.0),
        "tokens_per_second_median": float(inputs.shape[0] * inputs.shape[1] / statistics.median(samples)),
        "peak_memory_megabytes": float(peak),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Phase II score variants with synchronized timing.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2/timing"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "amp", "bf16", "fp16"], default="amp")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--variants", nargs="+", default=["D0", "R0", "DD", "DR", "RR"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-supports", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--parameter-match-target", type=int, default=None)
    args = parser.parse_args()
    set_seed(29)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq_len in args.seq_lens:
        inputs = torch.randint(0, 64, (args.batch_size, seq_len), device=device)
        for variant in args.variants:
            spec = {
                "model_name": variant,
                "task_type": "language",
                "vocab_size": 64,
                "max_seq_len": max(args.seq_lens),
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "num_supports": args.num_supports,
                "memory_output": "both",
                "parameter_match_target": args.parameter_match_target,
            }
            model = make_model(spec).to(device)
            row = {"variant": variant, "sequence_length": seq_len, "precision": args.precision, "backward": args.backward, "parameters": parameter_count(model), "approximate_flops": approximate_flops({**spec, "memory_score": variant in {"DD", "DR", "RR"}}, seq_len, batch_size=args.batch_size)}
            row.update(_time_one(model, inputs, device=device, iterations=args.iterations, warmup=args.warmup, backward=args.backward, precision=args.precision))
            rows.append(row)
            print(row, flush=True)
            del model
    with (args.output_dir / "timing.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save_json(args.output_dir / "timing.json", {"device": str(device), "settings": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
