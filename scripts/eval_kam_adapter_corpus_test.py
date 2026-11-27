#!/usr/bin/env python3
"""Score held-out raw continuation from the pinned wiki, code, and math streams."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from kam_adapter_preservation import ARMS, FACTORIAL_RECIPES, SEEDS, digest, key, layers_for, load_registered_model, model_identity
from kam_pretrained_adapters import AdapterStack, mixed_precision
from train_kam_adapter_preservation import DOMAINS, KL_STRIDE, LENGTH, corpus

CHUNKS = 64
BATCH = 4


@torch.inference_mode()
def score(model, tuner, frames):
    readouts = {}
    for domain, windows in frames.items():
        nll, kl = [], []
        for batch_index, segment in enumerate(windows.split(BATCH)):
            context = segment[:, :-1]
            phase = batch_index % KL_STRIDE
            target = segment[:, 1:]
            if tuner is not None:
                tuner.active = False
                with mixed_precision():
                    teacher = model.model(input_ids=context, use_cache=False).last_hidden_state
                    teacher_logp = model.lm_head(teacher[:, phase::KL_STRIDE]).float().log_softmax(-1)
                tuner.active = True
            with mixed_precision():
                state = model.model(input_ids=context, use_cache=False).last_hidden_state
                scores = model.lm_head(state)
                nll.extend(F.cross_entropy(scores.flatten(0, 1), target.reshape(-1),
                                           reduction="none").reshape_as(target).mean(-1).tolist())
                if tuner is not None:
                    student_logp = scores[:, phase::KL_STRIDE].float().log_softmax(-1)
                    kl.extend(F.kl_div(student_logp, teacher_logp, reduction="none",
                                       log_target=True).sum(-1).mean(-1).tolist())
        assert len(nll) == CHUNKS and (not kl or len(kl) == CHUNKS)
        readouts[domain] = {"nll_per_chunk": nll, "kl_per_chunk": kl if tuner is not None else None}
    return readouts


def evaluate(root, index, matched):
    model, _ = load_registered_model(root)
    model_id, revision, _ = model_identity(root)
    file = root / "token-manifest.json"
    manifest = json.loads(file.read_text())
    rng = torch.Generator().manual_seed(42053)
    frames, selected = {}, {}
    for domain, tokens in zip(DOMAINS, corpus(root, "test", manifest), strict=True):
        possible = tokens.numel() // (LENGTH + 1)
        assert possible >= CHUNKS
        blocks = torch.randperm(possible, generator=rng)[:CHUNKS]
        starts = blocks.cuda() * (LENGTH + 1)
        frames[domain] = tokens[starts[:, None] + torch.arange(LENGTH+1, device="cuda")].long()
        selected[domain] = blocks.tolist()

    if index == len(SEEDS):
        assert not matched
        runs = [(0, "untouched", "pretrained")]
    else:
        recipes = ("reference", "high_lr_kl") if matched else FACTORIAL_RECIPES
        runs = [(SEEDS[index], recipe, arm) for recipe in recipes for arm in ARMS]
    findings = {}
    for seed, recipe, arm in runs:
        name = key(seed, recipe, arm)
        checkpoint_hash = None
        scale = 1.0
        tuner = None
        try:
            if arm != "pretrained":
                path = root / f"adapter-{name}.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                spec = checkpoint["identity"]
                assert (spec["seed"], spec["recipe"], spec["arm"], spec["model"], spec["revision"]) == (
                    seed, recipe, arm, model_id, revision,
                )
                tuner = AdapterStack(model, arm, seed=seed, layers=layers_for(model))
                tuner.modules.load_state_dict(checkpoint["adapter"])
                tuner.modules.eval()
                checkpoint_hash = digest(path)
                if matched:
                    calibration = json.loads((root / f"matched-drift-{name}.json").read_text())
                    assert calibration["adapter_sha256"] == checkpoint_hash
                    scale = calibration["scale"]
                    assert 0 <= scale <= 1 and calibration["kl_cap"] == .01
                    assert any(row["scale"] == scale and row["forward_kl"] <= .01
                               for row in calibration["sampled_validation"])
                    with torch.no_grad():
                        for module in tuner.modules:
                            module.up.weight.mul_(scale)
            outputs = score(model, tuner, frames)
            findings[name] = {"adapter_sha256": checkpoint_hash, "domains": outputs, "scale": scale}
            print(model_id, name, flush=True)
        finally:
            if tuner is not None:
                tuner.close()
    payload = {"model": model_id, "revision": revision, "gpu": torch.cuda.get_device_name(),
               "token_manifest_sha256": digest(file), "chunk_indices": selected,
               "scored_tokens_per_chunk": LENGTH, "fits": findings}
    mode = "matched-" if matched else ""
    output = root / f"corpus-test-{mode}{index}.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, separators=(",", ":")))
    temp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int, choices=range(len(SEEDS)+1))
    parser.add_argument("--matched", action="store_true")
    args = parser.parse_args()
    evaluate(args.root, args.index, args.matched)
