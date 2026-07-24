from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch

from .factory import make_model
from .utils import choose_device, save_json, set_seed


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    iterations: int,
    warmup: int,
    backward: bool,
    device: torch.device,
) -> dict[str, float]:
    model.train(backward)
    for _ in range(warmup):
        output = model(inputs)
        if backward:
            output.float().square().mean().backward()
            model.zero_grad(set_to_none=True)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iterations):
        output = model(inputs)
        if backward:
            output.float().square().mean().backward()
            model.zero_grad(set_to_none=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    peak_bytes = (
        float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else float("nan")
    )
    batch_tokens = inputs.shape[0] * inputs.shape[1]
    return {
        "milliseconds_per_iteration": 1000.0 * elapsed / iterations,
        "tokens_per_second": batch_tokens * iterations / elapsed,
        "peak_memory_megabytes": peak_bytes / (1024.0**2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark KAM scaling against matched variants.")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-supports", type=int, default=64)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--backward", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, str, int | None]] = [
        ("kam-full", "kam", None),
        ("kam-local", "kam", args.local_window),
        ("kernel-self", "kernel-self", None),
        ("memory-only", "memory-only", None),
        ("dot-transformer", "dot-transformer", None),
    ]
    rows: list[dict[str, Any]] = []
    for sequence_length in args.seq_lens:
        inputs = torch.randint(
            0,
            args.vocab_size,
            (args.batch_size, sequence_length),
            device=device,
        )
        for variant_name, model_name, context_window in variants:
            spec = {
                "model_name": model_name,
                "task_type": "language",
                "vocab_size": args.vocab_size,
                "max_seq_len": max(args.seq_lens),
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "num_supports": args.num_supports,
                "context_window": context_window,
                "dropout": 0.0,
            }
            model = make_model(spec).to(device)
            result = time_model(
                model,
                inputs,
                iterations=args.iterations,
                warmup=args.warmup,
                backward=args.backward,
                device=device,
            )
            row = {
                "variant": variant_name,
                "sequence_length": sequence_length,
                "parameters": sum(p.numel() for p in model.parameters()),
                **result,
            }
            rows.append(row)
            print(row, flush=True)
            del model

    with (args.output / "timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        args.output / "timing.json",
        {
            "device": str(device),
            "backward": args.backward,
            "settings": vars(args),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
