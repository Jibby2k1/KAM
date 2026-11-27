#!/usr/bin/env python3
"""Time an exact cache of frozen teacher activations on the registered fit schedule."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from kam_adapter_preservation import ARMS, SEEDS, digest, layers_for, load_registered_model, model_identity
from kam_pretrained_adapters import AdapterStack, mixed_precision
from probe_kam_kl_prefix import backward, optimizer_trace
from train_kam_adapter_preservation import BATCHES, KL_STRIDE, LENGTH, STEPS, batch, corpus, layout, save, teacher_cache_identity


@torch.inference_mode()
def capture_teacher(model, samples):
    with mixed_precision():
        warm = model.model(input_ids=samples[0][:, :-1], use_cache=False).last_hidden_state
    assert warm.dtype == torch.bfloat16
    batch_size, length, hidden = warm.shape
    tape = torch.empty((len(samples), batch_size, length // KL_STRIDE, hidden),
                       dtype=warm.dtype, device=warm.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for index, packed in enumerate(samples):
        phase = index % KL_STRIDE
        with mixed_precision():
            state = model.model(input_ids=packed[:, :-1], use_cache=False).last_hidden_state
        tape[index].copy_(state[:, phase::KL_STRIDE])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    workspace = warm.new_empty(warm.shape)
    for phase, sample in enumerate(samples[:KL_STRIDE]):
        selected = workspace[:, phase::KL_STRIDE]
        selected.copy_(tape[phase])
        with mixed_precision():
            exact = model.model(input_ids=sample[:, :-1], use_cache=False).last_hidden_state
            exact_view = exact[:, phase::KL_STRIDE]
            assert selected.stride() == exact_view.stride()
            torch.testing.assert_close(selected, exact_view, rtol=0, atol=0)
            torch.testing.assert_close(model.lm_head(selected), model.lm_head(exact_view), rtol=0, atol=0)
    return tape, elapsed


def run(root, seed, write_tape):
    model, _ = load_registered_model(root)
    ident, revision, _ = model_identity(root)
    layers = layers_for(model)
    torch.use_deterministic_algorithms(True)
    manifest = root / "token-manifest.json"
    frozen = corpus(root, "train", json.loads(manifest.read_text()))
    windows = layout(frozen, STEPS, seed + 1)
    exposure = hashlib.sha256(b"".join(x.numpy().tobytes() for x in windows)).hexdigest()
    windows = [x.cuda() for x in windows]
    offsets = torch.arange(LENGTH + 1, device="cuda")
    samples = [batch(frozen, [indices[step] for indices in windows], offsets) for step in range(STEPS)]
    tape, build_s = capture_teacher(model, samples)
    print(ident, "teacher activation bytes", tape.numel() * tape.element_size(), "build seconds", build_s,
          flush=True)
    if write_tape is not None:
        started = time.perf_counter()
        cache_identity = teacher_cache_identity(root, model, seed, exposure)
        save(write_tape, {"identity": cache_identity, "activations": tape.cpu()})
        receipt = {
            "identity": cache_identity, "cache_build_s": build_s, "cache_write_s": time.perf_counter()-started,
            "file_bytes": write_tape.stat().st_size, "verified_teacher_phases": KL_STRIDE,
        }
        write_tape.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
        return
    report = {}
    for arm in ARMS:
        stack = AdapterStack(model, arm, seed=seed, layers=layers)
        try:
            initial = {name: x.detach().clone() for name, x in stack.modules.state_dict().items()}
            backward(model, stack, samples[0], 0)
            workspace = tape.new_empty((len(samples[0]), LENGTH, tape.size(-1)))
            workspace[:, ::KL_STRIDE].copy_(tape[0])
            backward(model, stack, samples[0], 0, teacher_hidden=workspace[:, ::KL_STRIDE])
            if arm == "kam":
                plain, original_ms = optimizer_trace(model, stack, samples, layers[0], initial, False)
                cached, cached_ms = optimizer_trace(model, stack, samples, layers[0], initial, False,
                                                    teacher_tape=tape)
            else:
                cached, cached_ms = optimizer_trace(model, stack, samples, layers[0], initial, False,
                                                    teacher_tape=tape)
                plain, original_ms = optimizer_trace(model, stack, samples, layers[0], initial, False)
            for a, b in zip(plain, cached, strict=True):
                assert a == b, ("teacher activation cache diverged", arm, a["step"])
            report[arm] = {"timing_order": ["original", "cached"] if arm == "kam" else ["cached", "original"],
                           "exact_optimizer_trajectory": True, "original_update_ms": original_ms,
                           "cached_update_ms": cached_ms, "trace": plain}
            print(ident, arm, "exact through", len(plain), "updates", flush=True)
        finally:
            stack.close()
    payload = {"model": ident, "revision": revision, "seed": seed, "steps": STEPS, "kl_stride": KL_STRIDE,
               "train_tokens": STEPS * LENGTH * sum(BATCHES), "gpu": torch.cuda.get_device_name(),
               "teacher_cache_bytes": tape.numel() * tape.element_size(), "dtype": str(tape.dtype),
               "vocab_size": model.config.vocab_size, "cache_build_s": build_s,
               "train_exposure_sha256": exposure, "token_manifest_sha256": digest(manifest),
               "exact_teacher_logits_phases": KL_STRIDE, "arms": report}
    output = root / "teacher-hidden-cache-256.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=731221)
    parser.add_argument("--write-tape", type=Path)
    args = parser.parse_args()
    run(args.root, args.seed, args.write_tape)
