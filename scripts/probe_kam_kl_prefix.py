#!/usr/bin/env python3
"""Verify and time shared frozen-prefix activations for forward KL."""
import argparse
import json
import math
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from statistics import median
from unittest.mock import patch

import torch

from kam_adapter_preservation import digest, key, layers_for, load_registered_model, model_identity
from kam_pretrained_adapters import AdapterStack
from train_kam_adapter_preservation import BATCHES, KL_STRIDE, LENGTH, STEPS, batch, corpus, layout, losses, optimizer_fingerprint


@contextmanager
def reuse_prefix(model, stack, last):
    captured = [None] * (last + 1)

    def reuse(index, forward):
        def run(*args, **kwargs):
            if not stack.active:
                output = forward(*args, **kwargs)
                if isinstance(output, tuple):
                    assert len(output) == 1
                    captured[index] = output if index == last else (None,)
                else:
                    assert isinstance(output, torch.Tensor)
                    captured[index] = output if index == last else None
                return output
            if index == last:
                output = captured[index]
                assert output is not None
                captured[index] = None
                return output
            hidden = args[0] if args else kwargs["hidden_states"]
            return (hidden,) if isinstance(captured[index], tuple) else hidden

        return run

    with ExitStack() as contexts:
        for index, layer in enumerate(model.model.layers[:last + 1]):
            contexts.enter_context(patch.object(layer, "forward", reuse(index, layer.forward)))
        yield


def backward(model, stack, packed, phase, *, teacher_hidden=None):
    stack.modules.zero_grad(set_to_none=True)
    ce, kl = losses(model, stack, packed, 4, phase, teacher_hidden=teacher_hidden)
    (ce + 4 * kl).backward()
    return torch.stack((ce.detach(), kl.detach()))


def compare(model, stack, packed, phase, first_layer):
    baseline = backward(model, stack, packed, phase)
    plain_grads = [p.grad.detach().clone() for p in stack.modules.parameters()]
    repeat = backward(model, stack, packed, phase)
    repeat_grads = [p.grad.detach().clone() for p in stack.modules.parameters()]
    with reuse_prefix(model, stack, first_layer):
        shared = backward(model, stack, packed, phase)
    torch.testing.assert_close(repeat, baseline, rtol=0, atol=0)
    torch.testing.assert_close(shared, baseline, rtol=0, atol=0)
    shared_grads = [p.grad.detach() for p in stack.modules.parameters()]
    return {"phase": phase, "ce": float(baseline[0]), "kl": float(baseline[1]),
            "plain_repeat": gradient_difference(plain_grads, repeat_grads),
            "shared_vs_plain": gradient_difference(plain_grads, shared_grads)}


def gradient_difference(reference, actual):
    differences = [a - b for a, b in zip(reference, actual, strict=True)]
    ref_norm = sum(float(grad.square().sum()) for grad in reference) ** .5
    error = sum(float(delta.square().sum()) for delta in differences) ** .5
    return {"max_abs": max(float(delta.abs().max()) for delta in differences),
            "relative_l2": error / ref_norm if ref_norm else None, "exact": error == 0}


def timed(model, stack, samples, first_layer, share):
    context = reuse_prefix(model, stack, first_layer) if share else nullcontext()
    stack.modules.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    with context:
        for phase, packed in enumerate(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            backward(model, stack, packed, phase % 4)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    return {"median_ms": median(times), "ms": times,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated()}


def optimizer_trace(model, stack, samples, first_layer, initial, share, *, teacher_tape=None):
    stack.modules.load_state_dict(initial)
    parameters = list(stack.modules.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=.01, fused=True)
    context = reuse_prefix(model, stack, first_layer) if share else nullcontext()
    trace, timings = [], []
    if teacher_tape is not None:
        workspace = teacher_tape.new_empty((len(samples[0]), LENGTH, teacher_tape.size(-1)))
        assert not share
    with context:
        for step, packed in enumerate(samples, start=1):
            progress = max(0, (step - 16) / (STEPS - 16))
            optimizer.param_groups[0]["lr"] = 8e-4 * min(step / 16, 1) * (.1 + .45 * (1 + math.cos(math.pi * progress)))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            phase = (step - 1) % KL_STRIDE
            selected = None
            if teacher_tape is not None:
                # Restore the same strided shape passed to the vocabulary head.
                selected = workspace[:, phase::KL_STRIDE]
                selected.copy_(teacher_tape[step-1])
            ce, kl = backward(model, stack, packed, phase, teacher_hidden=selected)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
            trace.append({"step": step, "ce": float(ce), "kl": float(kl),
                          "state_sha256": optimizer_fingerprint(parameters, optimizer)})
    return trace, timings


def replay_optimizer(root, output, arm, steps):
    model, _ = load_registered_model(root)
    model_id, revision, _ = model_identity(root)
    layers = layers_for(model)
    seed = 731221
    stack = AdapterStack(model, arm, seed=seed, layers=layers)
    initial = {name: value.detach().clone() for name, value in stack.modules.state_dict().items()}
    manifest = json.loads((root / "token-manifest.json").read_text())
    data = corpus(root, "train", manifest)
    windows = [positions.cuda() for positions in layout(data, STEPS, seed + 1)]
    offsets = torch.arange(LENGTH + 1, device="cuda")
    samples = [batch(data, [positions[index] for positions in windows], offsets) for index in range(steps)]
    torch.use_deterministic_algorithms(True)
    # Exclude initialization and first-kernel overhead from the timed traces.
    backward(model, stack, samples[0], 0)
    with reuse_prefix(model, stack, layers[0]):
        backward(model, stack, samples[0], 0)
    plain, plain_ms = optimizer_trace(model, stack, samples, layers[0], initial, False)
    shared, shared_ms = optimizer_trace(model, stack, samples, layers[0], initial, True)
    for original, optimized in zip(plain, shared, strict=True):
        assert original == optimized, ("optimizer state diverged", original["step"])
    result = {"steps": len(plain), "model": model_id, "revision": revision, "arm": arm, "seed": seed,
              "tokens_per_batch": LENGTH * sum(BATCHES), "gpu": torch.cuda.get_device_name(),
              "deterministic": True, "plain_ms": plain_ms, "shared_ms": shared_ms,
              "trace": plain, "exact_optimizer_replay": True}
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(output)
    print(model_id, arm, len(plain), median(plain_ms), median(shared_ms), flush=True)


def run(root, output, arm):
    model, _ = load_registered_model(root)
    ident, revision, _ = model_identity(root)
    layers = layers_for(model)
    seed = 731221
    stack = AdapterStack(model, arm, seed=seed, layers=layers)
    checkpoint = root / f"adapter-{key(seed, 'high_lr_kl', arm)}.pt"
    weights = torch.load(checkpoint, map_location="cuda", weights_only=True)
    identity = weights["identity"]
    assert (identity["model"], identity["revision"], identity["seed"], identity["recipe"], identity["arm"]) == (
        ident, revision, seed, "high_lr_kl", arm,
    )
    stack.modules.load_state_dict(weights["adapter"])
    del weights
    manifest = json.loads((root / "token-manifest.json").read_text())
    data = corpus(root, "validation", manifest)
    windows = [selection.cuda() for selection in layout(data, 12, 83011)]
    offsets = torch.arange(LENGTH + 1, device="cuda")
    samples = [batch(data, [p[i] for p in windows], offsets) for i in range(12)]
    equivalence = [compare(model, stack, packed, phase, layers[0])
                   for phase, packed in enumerate(samples[:4])]
    torch.use_deterministic_algorithms(True)
    try:
        deterministic = {"batches": [
            compare(model, stack, packed, phase, layers[0])
            for phase, packed in enumerate(samples[:4])
        ]}
    except RuntimeError as error:
        deterministic = {"error": str(error).splitlines()[0]}
    finally:
        torch.use_deterministic_algorithms(False)
    rounds = [timed(model, stack, samples, layers[0], share) for share in (False, True, True, False)]
    result = {"model": ident, "revision": revision, "adapter_sha256": digest(checkpoint),
              "seed": seed, "arm": arm, "tokens_per_batch": LENGTH * sum(BATCHES),
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "shared_layers": layers[0] + 1, "equivalence_normal": equivalence,
              "equivalence_deterministic": deterministic,
              "method_order": ["original", "shared", "shared", "original"], "rounds": rounds}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(ident, arm, [round(run["median_ms"], 2) for run in rounds], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", choices=("kam", "dot"), required=True)
    parser.add_argument("--optimizer-replay", action="store_true")
    parser.add_argument("--optimizer-steps", type=int, default=64)
    options = parser.parse_args()
    if options.optimizer_replay:
        if not 1 <= options.optimizer_steps <= STEPS:
            parser.error(f"--optimizer-steps must be between 1 and {STEPS}")
        replay_optimizer(options.root, options.output, options.arm, options.optimizer_steps)
    else:
        run(options.root, options.output, options.arm)
