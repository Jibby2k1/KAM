#!/usr/bin/env python3
"""Check tiled top-four fidelity and profile it against forced Flash SDPA."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kam_exact_sparse_readout import exact_sparse_read


def dense(q, k, v, causal):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal and q.size(-2) > 1)


def reference(q, k, v, causal):
    qlen, klen = q.size(-2), k.size(-2)
    score = (q.float() @ k.float().transpose(-1, -2)) * .25
    if causal:
        queries = torch.arange(qlen, device=q.device) + klen - qlen
        keys = torch.arange(klen, device=q.device)
        score = score.masked_fill(keys[None, :] > queries[:, None], -torch.inf)
    idx = score.argsort(dim=-1, descending=True, stable=True)[..., :4]
    top = score.gather(-1, idx)
    live = v.unsqueeze(-3).expand(-1, -1, qlen, -1, -1)
    live = live.gather(-2, idx[..., None].expand(*idx.shape, v.size(-1)))
    return (top.softmax(-1).unsqueeze(-1) * live.float()).sum(-2).bfloat16()


def profile(actions, repeats=30):
    """Measure warmed synchronized wall time and incremental allocated CUDA memory."""
    runs = {key: [] for key in actions}
    event = {key: [] for key in actions}
    names = list(actions)
    for action in actions.values():
        for _ in range(3):
            action()
    torch.cuda.synchronize()
    peaks = {}
    for name, action in actions.items():
        start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = action()
        torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() - start
        del out
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for index in range(repeats):
        for name in names[index % len(names):] + names[:index % len(names)]:
            start = perf_counter_ns()
            begin.record()
            out = actions[name]()
            end.record()
            end.synchronize()
            runs[name].append((perf_counter_ns() - start) / 1e6)
            event[name].append(begin.elapsed_time(end))
            del out
    return {name: {"wall_ms": runs[name], "cuda_ms": event[name], "incremental_allocated_bytes": peaks[name]}
            for name in names}


def tensors(batch, queries, keys):
    return (torch.randn(batch, 4, queries, 16, device="cuda", dtype=torch.bfloat16),
            *[torch.randn(batch, 4, keys, 16, device="cuda", dtype=torch.bfloat16) for _ in range(2)])


@torch.inference_mode()
def benchmark(output):
    import triton

    torch.manual_seed(460920)
    fidelity = {}
    for batch, nq, nk in ((1, 1, 1), (1, 67, 67), (2, 257, 257), (1, 1, 8192), (2, 17, 513)):
        q, k, v = tensors(batch, nq, nk)
        actual = exact_sparse_read(q, k, v, causal=True)
        target = reference(q, k, v, True)
        torch.testing.assert_close(actual, target, atol=0.016, rtol=0.01)
        fidelity[f"{batch}-{nq}-{nk}"] = {"max_abs_diff": float((actual.float() - target.float()).abs().max()),
                                            "n_values": actual.numel(), "exact_values": int(actual.eq(target).sum())}
    trials = {}
    for keys in (512, 2048, 4096, 8192):
        for kind, queries in (("prefill", keys), ("read-one", 1)):
            q, k, v = tensors(1, queries, keys)
            trials[f"{kind}-{keys}"] = profile({"flash": lambda: dense(q, k, v, True),
                                                "sparse": lambda: exact_sparse_read(q, k, v, causal=True)})
    output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                                  "triton": triton.__version__, "fidelity": fidelity, "trials": trials}, sort_keys=True)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    benchmark(args.output)
