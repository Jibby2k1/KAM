#!/usr/bin/env python3
"""Measure matched linear and SiLU query maps on fixed E5 first-hop keys."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_kam_natural_data import stable, words
from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from kam_musique_cached_retrieval import ENCODER, ENCODER_FILES, ENCODER_LIMIT, GPU, REVISION
from prepare_kam_musique_endpoint_pairs import candidate
from prepare_kam_native_musique import MEMBERS, canonical_paragraph
from prepare_kam_natural_router import title_key
from summarize_kam_pretrained_endpoint_selection import family_difference

PROTOCOL = "kam-query-activation-holdout-v1"
BUDGETS = (64, 256, 512)
EPOCHS = 6
WIDTH = 32
SCALE = 20.


def prepare(source, benchmark, encoder_manifest, work):
    cases = json.loads(benchmark.read_text())["cases"]
    held = {r["id"]: r["family"] for r in cases if r["cohort"] == "fresh_dev"}
    held_families = set(held.values())
    text = json.loads(encoder_manifest.read_text())

    def pack(raw, q):
        support = q["evidence"][0]
        distractors = {stable("\x1f".join(canonical_paragraph(p))): p["title"] + ". " + p["paragraph_text"]
                       for p in raw["paragraphs"] if not p["is_supporting"]}
        for evidence in q["evidence"]:
            distractors.pop(evidence["id"], None)
        order = sorted(distractors, key=lambda ident: stable(PROTOCOL + q["id"] + ident))[:16]
        if len(order) != 16:
            return None
        ids = [support["id"], *order]
        return {"id": q["id"], "family": stable(q["family"]), "question": q["question"], "pool_ids": ids,
                "passages": [support["title"] + ". " + support["body"], *[distractors[i] for i in order]]}

    with ZipFile(source) as archive:
        with archive.open(MEMBERS["dev"]) as stream:
            dev = [(raw, candidate(raw)) for raw in map(json.loads, stream) if raw["id"] in held]
        with archive.open(MEMBERS["train"]) as stream:
            train = sorted(map(json.loads, stream), key=lambda raw: stable(PROTOCOL + raw["id"]))
    unseen = [row for raw, q in dev if (row := pack(raw, q)) is not None]
    assert {x["id"]: x["family"] for x in unseen}.items() <= held.items() and len(unseen) >= 64
    reserved = {r["id"] for r in cases}
    used_questions = {tuple(words(q["question"])) for _, q in dev}
    used_titles = {title_key(d["title"]) for _, q in dev for d in q["evidence"]}
    eligible = []
    for raw in train:
        if raw["id"] in reserved:
            continue
        item = candidate(raw)
        if item and item["hops"] == 2 and stable(item["family"]) not in held_families:
            eligible.append((raw, item))
    fit, seen, fit_families = [], [], set()
    for raw, q in eligible:
        group = stable(q["family"])
        wording, titles = tuple(words(q["question"])), {title_key(d["title"]) for d in q["evidence"]}
        if wording in used_questions or titles & used_titles:
            continue
        packed = pack(raw, q)
        if packed is None:
            continue
        if len(fit) < BUDGETS[-1]:
            fit.append(packed)
            fit_families.add(group)
        elif group in fit_families:
            seen.append(packed)
        else:
            continue
        used_questions.add(wording)
        used_titles.update(titles)
        if len(seen) == 128:
            break
    assert len(fit) == BUDGETS[-1] and len(seen) >= 64
    data = {"fit": fit, "seen_family": seen, "unseen_family": unseen}
    work.mkdir(parents=True, exist_ok=True)
    panel = work / "panel.json"
    write_json(panel, data)
    write_json(work / "manifest.json", {"protocol": PROTOCOL, "panel_sha256": digest(panel),
               "source_sha256": digest(source), "reader_fixture_sha256": digest(benchmark),
               "encoder": ENCODER, "revision": REVISION, "encoder_files": text["encoder_files"],
               "budgets": BUDGETS, "epochs": EPOCHS, "bottleneck": WIDTH, "seeds": SEEDS,
               "cases": {split: [{k: row[k] for k in ("id", "family", "pool_ids")} for row in rows] for split, rows in data.items()}})


def evaluate(work, assets):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from run_kam_natural_feasibility import Encoder

    manifest, panel = work / "manifest.json", work / "panel.json"
    registration, data = [json.loads(p.read_text()) for p in (manifest, panel)]
    assert registration["protocol"] == PROTOCOL and registration["panel_sha256"] == digest(panel)
    assert registration["revision"] == json.loads((assets / "revisions.json").read_text())[ENCODER] == REVISION
    assert registration["encoder_files"] == {n: digest(assets / "encoder" / n) for n in ENCODER_FILES}
    assert torch.cuda.get_device_name() == GPU
    encoder = Encoder(assets / "encoder", batch=64)
    features = {}
    for split, rows in data.items():
        queries = encoder([r["question"] for r in rows], "query", max_length=ENCODER_LIMIT)
        keys = encoder([p for r in rows for p in r["passages"]], "passage", max_length=ENCODER_LIMIT)
        features[split] = queries.clone(), keys.view(len(rows), 17, -1).clone()
    del encoder

    def logits(q, k):
        return (k @ q.unsqueeze(-1)).squeeze(-1) * SCALE

    def measure(transform, split, limit=None):
        q, k = features[split]
        with torch.no_grad():
            logits_out = logits(transform(q[:limit]), k[:limit])
            target = torch.zeros(len(logits_out), device="cuda", dtype=torch.long)
            ranks = (logits_out[:, 1:] >= logits_out[:, :1]).sum(-1) + 1
            return {"nll": float(F.cross_entropy(logits_out, target)), "ranks": ranks.tolist(),
                    "recall_at_1": float(ranks.eq(1).float().mean()), "recall_at_3": float(ranks.le(3).float().mean())}

    modes = ("linear", "silu")
    results = {"untuned_e5": {split: measure(lambda q: q, split) for split in features}, "budgets": {}}
    q, keys = features["fit"]
    target = torch.zeros(len(q), device="cuda", dtype=torch.long)
    for budget in BUDGETS:
        arms = {mode: [] for mode in modes}
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            order = [i.cuda() for _ in range(EPOCHS) for i in torch.randperm(budget, generator=generator).split(32)]
            for mode in modes:
                torch.manual_seed(seed)
                inner = nn.Linear(q.shape[1], WIDTH, bias=False, device="cuda")
                outer = nn.Linear(WIDTH, q.shape[1], bias=False, device="cuda")
                nn.init.zeros_(outer.weight)
                optimizer = torch.optim.AdamW([*inner.parameters(), *outer.parameters()], lr=0.001, weight_decay=0.001)
                activation = F.silu if mode == "silu" else nn.Identity()

                def transform(query):
                    return F.normalize(query + outer(activation(inner(query))), dim=-1)

                for idx in order:
                    scores = logits(transform(q[idx]), keys[idx])
                    loss = F.cross_entropy(scores, target[idx])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                arms[mode].append({split: measure(transform, split, budget if split == "fit" else None) for split in features})
        contrasts = {}
        for split in ("seen_family", "unseen_family"):
            families = [r["family"] for r in data[split]]
            matrices = {mode: np.array([run[split]["ranks"] for run in arm]) for mode, arm in arms.items()}
            contrasts[split] = {f"recall_at_{k}": family_difference(matrices["silu"] <= k, matrices["linear"] <= k, families)
                                for k in (1, 3)}
        trained_families = {r["family"] for r in data["fit"][:budget]}
        results["budgets"][budget] = {"arms": arms, "silu_minus_linear": contrasts,
                                      "seen_questions_with_trained_family": sum(r["family"] in trained_families
                                                                                for r in data["seen_family"])}
    write_json(work / "results.json", {"protocol": PROTOCOL, "manifest_sha256": digest(manifest), "gpu": GPU, **results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("source", "benchmark", "encoder_manifest", "work"):
        prep.add_argument(name, type=Path)
    run = commands.add_parser("evaluate")
    run.add_argument("work", type=Path)
    run.add_argument("assets", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.benchmark, args.encoder_manifest, args.work)
    else:
        evaluate(args.work, args.assets)
