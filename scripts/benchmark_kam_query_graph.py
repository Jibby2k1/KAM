#!/usr/bin/env python3
"""Profile CUDA graph prefill for Flash and protected/windowed attention on HPG."""
import argparse
from contextlib import contextmanager
import gc
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS

from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import load
from kam_adapter_preservation import SEEDS, digest
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_pretrained_adapters import mixed_precision
from kam_query_window_flash import WindowAttention

ARMS = ("flash", "protected", "tail-256")
SHAPES = ((1, 4096), (1, 8192), (2, 4096), (2, 8192), (4, 4096), (4, 8192))


@contextmanager
def unpadded_dispatch(model, attention):
    """Keep the synthetic padding mask out of captured, strictly causal prefill."""
    name = "kam_query_graph_unpadded"
    original = model.config._attn_implementation
    assert original == "sdpa" and name not in ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[name] = attention.forward
    model.config._attn_implementation = name
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            yield
    finally:
        model.config._attn_implementation = original
        del ALL_ATTENTION_FUNCTIONS[name]


def prefill(model, stack, projection, attention, mode, tokens):
    attention.mode = mode
    stack.context(None, False)
    with mixed_precision():
        hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state
        return F.linear(hidden[:, -1], projection).float()


def capture(action):
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = action()
    torch.cuda.synchronize()
    return graph, output, {"capture_ms": (perf_counter()-start)*1000,
        "capture_peak_bytes": torch.cuda.max_memory_allocated()-baseline_allocated,
        "graph_live_bytes": torch.cuda.memory_allocated()-baseline_allocated}


def shape_report(model, stack, projection, attention, data, batch, width, repeats):
    sample = [[token for seq in book["windows"] for token in seq[:-1]][:width] for book in data["pg19"][:batch]]
    tokens = torch.tensor(sample, device="cuda")
    tensor_actions = {mode: lambda mode=mode: prefill(model, stack, projection, attention, mode, tokens) for mode in ARMS}
    receipts, graphs = {}, {}
    with attention.installed():
        standard = {mode: action() for mode, action in tensor_actions.items()}
    with unpadded_dispatch(model, attention):
        for mode, action in tensor_actions.items():
            stack.context(None, False)
            warm_stream = torch.cuda.Stream()
            warm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm_stream):
                for _ in range(3):
                    action()
            torch.cuda.current_stream().wait_stream(warm_stream)
            control = action()
            torch.testing.assert_close(control, standard[mode], rtol=0, atol=0)
            graph, output, receipt = capture(action)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, control, rtol=0, atol=.016)
            receipts[mode] = receipt | {"max_abs_logit_delta": float((output-control).abs().max()),
                                       "bitwise_parity": bool(torch.equal(output, control))}
            graphs[mode] = (graph, output)

        original = tokens.clone()
        tokens.copy_(original.roll(17, dims=-1))
        for mode, (graph, output) in graphs.items():
            control = tensor_actions[mode]()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, control, rtol=0, atol=.016)
            receipts[mode]["updated_input_max_abs_delta"] = float((output-control).abs().max())
        tokens.copy_(original)

        def replay(mode):
            graph, output = graphs[mode]
            graph.replay()
            return output

        actions = {f"eager/{mode}": action for mode, action in tensor_actions.items()}
        actions.update({f"graph/{mode}": lambda mode=mode: replay(mode) for mode in ARMS})
        results = profile(actions, repeats=repeats)
    for graph, _ in graphs.values():
        graph.reset()
    return {"timings": results, "capture": receipts}


@torch.inference_mode()
def evaluate(root, source, work, smoke):
    assert torch.cuda.get_device_name() == GPU
    seed = SEEDS[0]
    model, stack, projection, data, checkpoint, attention = load(root, source, work, seed, WindowAttention)
    repeats = 3 if smoke else 12
    results = {}
    for batch, width in SHAPES[:1] if smoke else SHAPES:
        results[f"{batch}x{width}"] = shape_report(model, stack, projection, attention, data, batch, width, repeats)
        gc.collect()
        torch.cuda.empty_cache()
    output = "graph-smoke.json" if smoke else "graph-profile.json"
    write_json(work / output, {"seed": seed, "gpu": GPU, "torch": str(torch.__version__),
        "checkpoint_sha256": checkpoint, "manifest_sha256": digest(work / "manifest.json"), "shapes": results})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "work"):
        p.add_argument(name, type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    evaluate(a.root, a.source, a.work, a.smoke)
