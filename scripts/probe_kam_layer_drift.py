#!/usr/bin/env python3
"""Measure single-layer and leave-one-out drift of low-rate residual adapters."""
import argparse
import json
from pathlib import Path

import torch

from eval_kam_adapter_preservation import load_checkpoint
from kam_adapter_preservation import ARMS, SEEDS, key
from kam_pretrained_adapters import mixed_precision
from train_kam_adapter_preservation import BATCHES, DOMAINS, LENGTH, batch, corpus, layout, validation


@torch.inference_mode()
def activation_ratios(model, stack, data, offsets):
    observations = [[] for _ in stack.modules]
    handles = []
    for index, module in enumerate(stack.modules):
        def measure(_, inputs, output, *, pos=index):
            x = inputs[0].float()
            delta = output.float() - x
            values = dict(zip(DOMAINS, (float(d.norm() / a.norm())
                                       for d, a in zip(delta.split(BATCHES), x.split(BATCHES), strict=True)),
                              strict=True))
            observations[pos].append(values)
        handles.append(module.register_forward_hook(measure))
    schedule = [p.cuda() for p in layout(data, 4, 83011)]
    try:
        with mixed_precision():
            for index in range(4):
                sample = batch(data, [p[index] for p in schedule], offsets)
                model.model(input_ids=sample[:, :-1], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert all(len(values) == 4 for values in observations)
    return observations


def probe(root, index):
    seed, arm = SEEDS[index // len(ARMS)], ARMS[index % len(ARMS)]
    model, _, stack, checkpoint = load_checkpoint(root, seed, "lower_lr", arm)
    manifest = root / "token-manifest.json"
    data = corpus(root, "validation", json.loads(manifest.read_text()))
    offsets = torch.arange(LENGTH+1, device="cuda")
    names = checkpoint["training"]["layers"]
    original = [t.up.weight.detach().clone() for t in stack.modules]
    ratios = activation_ratios(model, stack, data, offsets)
    conditions = {"full": set(names), "untouched": set()}
    for layer in names:
        conditions[f"only-{layer}"] = {layer}
        conditions[f"without-{layer}"] = set(names) - {layer}
    outcomes = {}
    for case, active in conditions.items():
        with torch.no_grad():
            for layer, module, saved in zip(names, stack.modules, original, strict=True):
                if layer in active:
                    module.up.weight.copy_(saved)
                else:
                    module.up.weight.zero_()
        outcomes[case] = {"active_layers": sorted(active), **validation(model, stack, data, offsets)}
    assert abs(outcomes["untouched"]["forward_kl"]) < 1e-6
    target = root / f"layer-drift-{key(seed, 'lower_lr', arm)}.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps({"checkpoint": checkpoint, "gpu": torch.cuda.get_device_name(),
                                "activation_ratio_by_layer": dict(zip(names, ratios, strict=True)),
                                "conditions": outcomes}, indent=2) + "\n")
    temp.replace(target)
    print(checkpoint["model"], seed, arm, outcomes["full"]["forward_kl"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=int, choices=range(len(ARMS) * len(SEEDS)))
    args = parser.parse_args()
    probe(args.root, args.index)
