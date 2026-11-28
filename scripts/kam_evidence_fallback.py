#!/usr/bin/env python3
"""Measure distant-evidence attention and evaluate a development-selected Flash layer mask."""
import argparse
import json
from pathlib import Path

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json

REGIONS = ("earlier", "earlier_outside_first", "bridge", "bridge_outside_first", "gold", "foil")
QUERIES = ("gold_end", "choice")
LAYERS = 32
HEADS = 32
N_FLASH = 16


def read(path):
    return json.loads(path.read_text())


def identities(rows):
    return [[x["id"], x["family"], x["expected"]] for x in rows]


def select(work):
    import numpy as np

    spans = work / "spans.json"
    labels = read(spans)["dev"]
    pieces, hashes = [], {}
    for seed in SEEDS:
        path = work / f"dev-mass-{seed}.json"
        receipt = read(path)
        assert receipt["spans_sha256"] == digest(spans) and receipt["seed"] == seed
        assert receipt["presentations"] == identities(labels)
        mass = np.array(receipt["sum_mass"], dtype=np.float64)
        assert mass.shape == (LAYERS, HEADS, len(QUERIES), len(REGIONS), 2)
        assert np.isfinite(mass).all() and (mass[..., 0] + .001 >= mass[..., 1]).all()
        idx = REGIONS.index("earlier_outside_first")
        pieces.append((mass[..., idx, 0] - mass[..., idx, 1]).mean(axis=(1, 2)) / len(labels))
        hashes[path.stem] = digest(path)
    merit = np.mean(pieces, axis=0).tolist()
    order = sorted(range(LAYERS), key=lambda i: (-merit[i], i))
    flash = sorted(order[:N_FLASH])
    write_json(work / "selection.json", {"spans_sha256": digest(spans), "dev_mass_sha256": hashes,
        "objective": "mean omitted attention mass on earlier evidence outside block zero",
        "dev_presentations": len(labels), "queries": QUERIES, "mean_omitted_mass_by_layer": merit,
        "ranked_layers_zero_based": order, "flash_layers_zero_based": flash,
        "shortlist_layers_zero_based": sorted(set(range(LAYERS))-set(flash))})


def runner(root, source, work, mode, index):
    import torch

    from benchmark_kam_sparse_live_read import profile
    from eval_kam_backbone_shortlist import BackboneAttention, WIDTHS, load
    from eval_kam_long_sparse import language_score, retrieval_score
    from eval_kam_pretrained import next_logits
    from kam_backbone_block_attention import BLOCK, QUERY_TILE, block_routes
    from kam_live_token_factorial import GPU
    from kam_pretrained_adapters import mixed_precision

    class EvidenceAttention(BackboneAttention):
        def begin_prompt(self, r, device):
            assert r["gold"][1]-1 >= 512
            self.rows = torch.tensor([r["gold"][1]-1, 4095], device=device)
            self.key_index = torch.arange(4096, device=device)
            self.blocks = self.key_index // BLOCK
            self.visible = self.key_index <= self.rows[:, None]
            ranges = {"earlier_outside_first": (max(BLOCK, r["earlier"][0]), r["earlier"][1]),
                      "bridge_outside_first": (max(BLOCK, r["bridge"][0]), r["bridge"][1]),
                      **{name: r[name] for name in ("earlier", "bridge", "gold", "foil")}}
            self.region = torch.stack([(self.key_index >= ranges[name][0]) &
                                       (self.key_index < ranges[name][1]) for name in REGIONS]).float()

        def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
            assert mask is None and dropout == 0 and not self.selected and scaling is not None
            _, heads, width, dim = q.shape
            kv = k.size(1)
            assert heads == HEADS and width == 4096
            chosen = block_routes(q, k, keep_first=True).index_select(-2, self.rows // QUERY_TILE)
            retained = (chosen[..., None] == self.blocks).any(dim=-2)
            self.bad_routes += (~(chosen == 0).any(dim=-1)).sum()
            with torch.autocast("cuda", enabled=False):
                grouped = q.index_select(-2, self.rows).float().reshape(1, kv, -1, dim)
                scores = (grouped @ k.float().transpose(-1, -2)).reshape(1, heads, len(QUERIES), width) * scaling
                dense = scores.masked_fill(~self.visible, -torch.inf).softmax(dim=-1)
                mass = torch.stack((dense, dense * retained), dim=-2)
                value = torch.einsum("bhqsk,rk->bhqrs", mass, self.region)[0]
            self.total[module.layer_idx] += value
            self.visits[module.layer_idx] += 1
            return super().forward(module, q, k, v, mask, dropout=dropout, scaling=scaling, **kwargs)

    span_file = work / "spans.json"
    spans = read(span_file)
    manifest_hash = digest(work / "manifest.json")
    assert spans["holdout_manifest_sha256"] == manifest_hash
    policy = work / "selection.json"
    seed = SEEDS[index] if index is not None else SEEDS[0]
    audit_mode = mode in ("dev", "holdout")
    model, stack, projection, data, checkpoint, attention = load(
        root, source, work, seed, EvidenceAttention if audit_mode else BackboneAttention)
    common = {"gpu": GPU, "seed": seed, "checkpoint_sha256": checkpoint, "manifest_sha256": manifest_hash,
              "spans_sha256": digest(span_file)}
    if audit_mode:
        prompts = data["retrieval"]
        if mode == "dev":
            path = work / "dev-inputs.json"
            assert spans["dev_inputs_sha256"] == digest(path)
            prompts = read(path)
        annotations = spans[mode]
        assert identities(prompts) == identities(annotations)
        if mode == "holdout":
            assert read(policy)["spans_sha256"] == digest(span_file)
        attention.total = torch.zeros((LAYERS, HEADS, len(QUERIES), len(REGIONS), 2), device="cuda", dtype=torch.float64)
        attention.bad_routes = torch.zeros((), device="cuda", dtype=torch.int64)
        attention.visits = [0] * LAYERS
        with torch.inference_mode(), attention.installed(), mixed_precision():
            stack.context(None, False)
            for prompt, r in zip(prompts, annotations, strict=True):
                attention.begin_prompt(r, device="cuda")
                model.model(input_ids=torch.tensor([prompt["ids"]], device="cuda"), use_cache=False)
        assert attention.visits == [len(annotations)] * LAYERS and int(attention.bad_routes) == 0
        write_json(work / f"{mode}-mass-{seed}.json", common | {
            "queries": QUERIES, "regions": REGIONS, "presentations": identities(annotations),
            "sum_mass": attention.total.tolist(), "selection_sha256": digest(policy) if mode == "holdout" else None})
        return

    selection = read(policy)
    assert selection["spans_sha256"] == digest(span_file)
    layers = frozenset(selection["shortlist_layers_zero_based"])
    assert len(layers) == LAYERS-N_FLASH
    attention.keep_first = True
    common["selection_sha256"] = digest(policy)
    if mode == "evaluate":
        attention.selected = layers
        with attention.installed():
            scores = {"pg19": language_score(model, stack, data),
                      "retrieval": retrieval_score(model, stack, projection, data)}
        write_json(work / f"fallback-{seed}.json", common | {"shortlist_layers": sorted(layers), **scores})
        return

    modes = {"flash": frozenset(), "periodic-16": frozenset(range(1, LAYERS, 2)),
             "selected-16": layers, "protected-32": frozenset(range(LAYERS))}
    sample = [token for seq in data["pg19"][0]["windows"] for token in seq[:-1]]
    timings = {}
    with attention.installed():
        for width in WIDTHS:
            tokens = torch.tensor([sample[:width]], device="cuda")
            mask = torch.ones_like(tokens, dtype=torch.bool)

            def action(selected):
                attention.selected = selected
                return next_logits(model, stack, tokens, mask, projection=projection)[0]

            for active in modes.values():
                assert torch.isfinite(action(active)).all()
            timings[width] = profile({name: lambda active=active: action(active) for name, active in modes.items()}, repeats=15)
    write_json(work / "fallback-profile.json", common | {"torch": str(torch.__version__),
        "modes": {name: sorted(active) for name, active in modes.items()}, "full_prefill": timings})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "work"):
        p.add_argument(name, type=Path)
    p.add_argument("mode", choices=("dev", "select", "holdout", "evaluate", "benchmark"))
    p.add_argument("index", type=int, nargs="?", choices=range(len(SEEDS)))
    a = p.parse_args()
    if a.mode == "select":
        select(a.work)
    else:
        runner(a.root, a.source, a.work, a.mode, a.index)
