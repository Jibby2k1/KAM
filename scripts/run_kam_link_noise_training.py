#!/usr/bin/env python3
"""Fit noise-aware pair heads with matched clean and sham controls."""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch.nn.functional import cross_entropy

from audit_kam_natural_data import digest, stable, words
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_fullwiki_support_links import ContextScorer, relation_features
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import prepared
from run_kam_link_noise import challenge, modified
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts

TEMPLATES = (
    "The catalogue references {title}.",
    "The index points to {title}.",
    "This entry cross-references {title}.",
    "The bibliography lists {title}.",
)
TEST_TEMPLATE = "For further reading, consult {title}."
FORMATS = ("clean", "sham", "linked")
TREATMENTS = {
    "repeated_clean": ("clean", True),
    "sham_trained": ("sham", True),
    "link_noise_trained": ("linked", True),
    "link_noise_without_graph": ("linked", False),
}


@torch.no_grad()
def materialize(a, cfg, fixture, records):
    cases = {row["id"]: row for row in fixture["rows"]["training"]}
    assert set(cases) == {r["id"] for r in records["rows"]}
    index = FullWikiIndex(a.inputs, read(a.inputs / "manifest.json"), cfg)
    enc = Encoder(a.assets / "encoder", 64)
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None,
                                    weights_sha256=a.baseline_sha256)
    queries = enc([cases[r["id"]]["question"] for r in records["rows"]], "query")
    depth = fixture["training_depth"]
    cpu = torch.triu_indices(depth, depth, 1)
    pairs = cpu.cuda()
    slot = {tuple(p): i for i, p in enumerate(cpu.T.tolist())}
    inputs, changed_docs, audit = [], [], []
    for i, row in enumerate(records["rows"]):
        source = cases[row["id"]]
        ranks = row["row_ids"]
        gold = tuple(p for p, q in enumerate(ranks) if q in source["gold_ids"])
        example = {**index.candidates(source, ranks), "gold": gold}
        graph, lexical = relation_features(example, cpu)
        keys = index.bank[torch.tensor(ranks, device="cuda")].unsqueeze(0)
        query = adapter(queries[i:i+1])
        data = challenge(example, graph, (query @ keys[0].T)[0])
        if data is None:
            continue
        donor, target, title, fake = data
        template = TEMPLATES[int(stable(row["id"])[:7], 16) % len(TEMPLATES)]
        variants = [modified(example, donor, ref, template) for ref in (fake, title)]
        assert all(variants[0]["context"][i] == variants[1]["context"][i] == example["context"][i]
                   for i in gold)
        assert len(words(variants[0]["context"][donor][1][0])) == len(words(variants[1]["context"][donor][1][0]))
        timed_variants = [relation_features(variant, cpu) for variant in variants]
        if timed_variants[0][0][donor, target] or not timed_variants[1][0][donor, target]:
            continue
        changed_docs.extend(candidate_texts(variant)[donor] for variant in variants)
        inputs.append((query, keys, donor, slot[gold], ((graph, lexical), *timed_variants)))
        audit.append({"id": row["id"], "donor": donor, "target": target, "label": slot[gold],
                      "template": template})
    assert len(inputs) >= 900
    updated = enc(changed_docs, "passage")
    assert len(updated) == 2 * len(inputs)
    xx, targets = {format_: [[], []] for format_ in FORMATS}, []
    for i, (query, keys, donor, label, variants) in enumerate(inputs):
        for j, (mode, (graph, lexical)) in enumerate(zip(FORMATS, variants, strict=True)):
            variant_keys = keys
            if j:
                variant_keys = keys.clone()
                variant_keys[0, donor] = updated[2 * i + j - 1]
            ref, x, _ = prepared(query, variant_keys, graph.cuda().unsqueeze(0), pairs)
            xx[mode][0].append(ref[0])
            xx[mode][1].append(torch.cat((x[0], lexical.cuda()), -1))
        targets.append(label)
    return {k: tuple(torch.stack(part) for part in v) for k, v in xx.items()}, torch.tensor(targets, device="cuda"), audit


def train(cfg, x, labels):
    found = {}
    for name, (companion, use_graph) in TREATMENTS.items():
        torch.manual_seed(51407)
        model = ContextScorer().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["pair_learning_rate"], fused=True)
        rng = torch.Generator(device="cuda").manual_seed(19113)
        mask = [use_graph or col not in (3, 4, 5, 9, 10) for col in range(11)]
        selector = torch.tensor(mask, device="cuda")
        clean_ref, clean_x = x["clean"]
        noise_ref, noise_x = x[companion]
        for _ in range(cfg["pair_epochs"]):
            for batch in torch.randperm(len(labels), generator=rng, device="cuda").split(cfg["batch"]):
                base = torch.cat((clean_ref[batch], noise_ref[batch]))
                features = torch.cat((clean_x[batch], noise_x[batch])) * selector
                loss = cross_entropy(model(base, features), labels[batch].repeat(2))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        assert torch.isfinite(loss)
        found[name] = {"weights": model.cpu().state_dict(), "mask": mask, "loss": float(loss.detach())}
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "config", "inputs", "fixture", "fit-records", "baseline", "output", "weights"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    a = p.parse_args()
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    cfg, fixture, fit_records = read(a.config), read(a.fixture), read(a.fit_records)
    assert fit_records["fixture_sha256"] == digest(a.fixture)
    assert fit_records["config_sha256"] == digest(a.config)
    assert fit_records["weights_sha256"] == a.baseline_sha256
    x, labels, records = materialize(a, cfg, fixture, fit_records)
    heads = train(cfg, x, labels)
    meta = {"fixture_sha256": digest(a.fixture), "config_sha256": digest(a.config),
            "baseline_sha256": a.baseline_sha256, "fit_records_sha256": digest(a.fit_records),
            "templates": TEMPLATES, "prespecified_test_template": TEST_TEMPLATE, "torch": str(torch.__version__)}
    torch.save({**meta, "arms": heads}, a.weights)
    save(a.output, {**meta, "gpu": torch.cuda.get_device_name(), "elapsed_seconds": perf_counter() - start,
                    "training_cases": records, "weights_sha256": digest(a.weights),
                    "training_results": {name: {"loss": item["loss"], "mask": item["mask"]}
                                         for name, item in heads.items()}})


if __name__ == "__main__":
    main()
