#!/usr/bin/env python3
"""Measure physical context compaction with original relative lag indices."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter

import torch

from run_kam_bitemporal_decoder import WIDTH, make_decoder

ARMS = ("dot_without_absolute", "radial_without_absolute")


class ContextExperiment:
    def __init__(self, arm, device, seed):
        self.masked = make_decoder(arm, seed, device).eval()
        generator = torch.Generator().manual_seed(seed + 1)
        for block in self.masked.blocks:
            assert block.context.window is None
            bias = block.context.relative_bias.bias
            # Nonzero relative lag terms exercise the original-position invariant.
            bias.detach().copy_(torch.randn(bias.shape, generator=generator).to(device) * .35)
        self.compact = deepcopy(self.masked)
        self.keep = None
        self.lag = None
        for block in self.masked.blocks:
            block.context.score.register_forward_hook(self.mask_score)
        for block in self.compact.blocks:
            lag_bias = block.context.relative_bias
            block.context.relative_bias = None
            self.attach_lag(block.context.score, lag_bias)
        self.max_lag = self.masked.blocks[0].context.relative_bias.max_lag
        assert all(b.context.relative_bias.max_lag == self.max_lag for b in self.masked.blocks)

    def mask_score(self, _module, _args, scores):
        return scores.masked_fill(~self.keep[:, None, None, :], torch.finfo(scores.dtype).min)

    def attach_lag(self, scorer, lag_bias):
        def add_bias(_module, _args, scores):
            return scores + lag_bias.bias[:, self.lag].permute(1, 0, 2, 3)
        scorer.register_forward_hook(add_bias)

    def packed(self, tokens, positions):
        x = tokens.gather(1, positions[..., None].expand(-1, -1, WIDTH))
        lag = (positions[:, :, None] - positions[:, None, :]).clamp(0, self.max_lag - 1)
        return x, lag

    def compact_forward(self, x, lag):
        self.lag = lag
        return self.compact(x)

    def pack_and_forward(self, tokens, positions):
        return self.compact_forward(*self.packed(tokens, positions))


def timings(run, barrier, iterations):
    for _ in range(4):
        run()
    barrier()
    ms = []
    for _ in range(iterations):
        start = perf_counter()
        run()
        barrier()
        ms.append((perf_counter() - start) * 1000)
    return {"p50_ms": median(ms), "p95_ms": quantiles(ms, n=100, method="inclusive")[94]}


@torch.inference_mode()
def measure(arm, device, steps, smoke=False):
    experiment = ContextExperiment(arm, device, 128719)
    rng = torch.Generator().manual_seed(7504)
    barrier = torch.cuda.synchronize if device == "cuda" else lambda: None
    rows = []
    for length in ((33, 97) if smoke else (97, 257)):
        for batch in ((1, 3) if smoke else (1, 64)):
            tokens = torch.randn((batch, length, WIDTH), generator=rng).to(device)
            order = torch.rand((batch, length - 1), generator=rng).argsort(1)
            for pi, fraction in enumerate((1., .94, .75, .5, .25)):
                count = max(2, round(fraction * length))
                positions = torch.cat((order[:, :count - 1].sort(1).values,
                                       torch.full((batch, 1), length - 1)), dim=1).to(device)
                keep = torch.zeros((batch, length), dtype=torch.bool, device=device)
                keep.scatter_(1, positions, True)
                experiment.keep = keep
                compact, lag = experiment.packed(tokens, positions)
                query = experiment.masked(tokens)
                packed_query = experiment.compact_forward(compact, lag)
                # Check both numeric outputs and the model's two stage decisions.
                torch.testing.assert_close(query, packed_query, rtol=2e-5, atol=5e-6)
                assert torch.equal(query[:, :64].argmax(1), packed_query[:, :64].argmax(1))
                assert torch.equal(query[:, 64:].argmax(1), packed_query[:, 64:].argmax(1))
                calls = [
                    ("masked_full", lambda: experiment.masked(tokens)),
                    ("compact_inputs_ready", lambda: experiment.compact_forward(compact, lag)),
                    ("compact_with_gather_and_lags", lambda: experiment.pack_and_forward(tokens, positions)),
                ]
                if pi % 2:
                    calls.reverse()
                rows.append({"batch": batch, "full_tokens": length, "active_tokens": count,
                             "max_abs_logit_delta": float((query - packed_query).abs().max()),
                             "timings": {k: timings(call, barrier, steps) for k, call in calls}})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.steps < 2:
        p.error("Use at least two timing repetitions")
    torch.set_num_threads(4 if a.device == "cuda" else 2)
    torch.set_float32_matmul_precision("highest")
    data = {"arm": a.arm, "device": torch.cuda.get_device_name() if a.device == "cuda" else "cpu",
            "torch": str(torch.__version__), "timing_iterations": a.steps,
            "rows": measure(a.arm, a.device, a.steps, a.smoke)}
    a.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
