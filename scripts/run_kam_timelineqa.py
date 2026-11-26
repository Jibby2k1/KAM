#!/usr/bin/env python3
"""Measure how external date provenance changes lookup on generated lifelogs."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from prepare_kam_timelineqa import DATE, SCHEMA
from run_kam_hotpot_independent import read, save
from run_kam_natural_feasibility import Encoder


def date_text(day, text):
    return f"Journal entry date {day}. {text}"


@torch.inference_mode()
def evaluate(persona, encoder):
    episodes, cases = persona["episodes"], persona["cases"]
    index = {k: i for i, k in enumerate(episodes)}
    texts = [date_text(r["day"], r["text"]) for r in episodes.values()]
    clean_vectors = encoder(texts, "passage")
    days = [DATE.search(r["question"])[0] for r in cases]
    changed = [date_text(day, episodes[r["donor"]]["text"]) for day, r in zip(days, cases, strict=True)]
    corrupted_vectors = encoder(changed, "passage")
    queries = encoder([r["question"] for r in cases], "query")
    outputs = []
    for row, day, query, corrupted in zip(cases, days, queries, corrupted_vectors, strict=True):
        assert episodes[row["gold"]]["day"] == day and episodes[row["donor"]]["day"] != day
        positions = torch.tensor([index[x] for x in row["candidates"]], device=query.device)
        semantic = clean_vectors[positions] @ query
        donor_slot = row["candidates"].index(row["donor"])
        alternate = semantic.clone()
        alternate[donor_slot] = corrupted @ query
        metadata = [episodes[k]["day"] == day for k in row["candidates"]]
        claimed = metadata.copy()
        claimed[donor_slot] = True
        target = row["candidates"].index(row["gold"])
        reference = set(i for i, is_day in enumerate(metadata) if is_day)
        assert target in reference and max(reference | {donor_slot}) < 16
        masks = [torch.tensor(m, device=query.device) for m in (metadata, claimed)]
        # Pool depth clips the original candidate list, but ranking tie order
        # is frozen independently of the gold and candidate depth.
        priority = torch.tensor(row["shuffle"], device=query.device)
        depth_results = {}
        for n in (16, 64, 256):
            noise = {}
            tie_order = priority[:n].argsort()
            ones = torch.ones(n, dtype=torch.bool, device=query.device)
            for variant, base, text_mask in (("true_date", semantic, masks[0]),
                                             ("misdated_noise", alternate, masks[1])):
                rankings = {}
                for name, allowed in (("dense", ones), ("reported_date", text_mask[:n]),
                                      ("authoritative_date", masks[0][:n])):
                    ranking = base[:n].masked_fill(~allowed, -torch.inf)[tie_order].argsort(descending=True, stable=True)
                    ranked = tie_order[ranking[:min(int(allowed.sum()), 2)]].tolist()
                    rankings[name] = {"support": ranked[0] == target, "support_top2": target in ranked,
                                      "wrong_day_top1": ranked[0] not in reference,
                                      "donor_top1": ranked[0] == donor_slot,
                                      "retrieved": [row["candidates"][i] for i in ranked]}
                noise[variant] = rankings
            assert noise["true_date"]["authoritative_date"] == noise["misdated_noise"]["authoritative_date"]
            depth_results[n] = noise
        outputs.append({"id": row["id"], "persona": persona["seed"], "density": persona["density"],
                        "category": episodes[row["gold"]]["type"], "n_same_day": len(reference),
                        "candidate_sha256": digest_candidates(row["candidates"]), "scores": depth_results})
    return outputs


def digest_candidates(candidates):
    from hashlib import sha256
    return sha256("\n".join(candidates).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "fixture", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    cfg = p.parse_args()
    assert 0 <= cfg.shard < cfg.shards
    torch.set_float32_matmul_precision("highest")
    start = perf_counter()
    fixture = read(cfg.fixture)
    assert fixture["schema"] == SCHEMA
    encoder = Encoder(cfg.assets / "encoder", 96)
    members = fixture["personas"][cfg.shard::cfg.shards]
    rows = [row for persona in members for row in evaluate(persona, encoder)]
    save(cfg.output, {"fixture_sha256": digest(cfg.fixture), "models": read(cfg.assets / "revisions.json"),
                      "author_generator_revision": fixture["author_generator_revision"],
                      "sources": [{k: r[k] for k in ("seed", "density", "original_sha256", "original_episodes")}
                                  for r in members],
                      "gpu": torch.cuda.get_device_name(), "elapsed_seconds": perf_counter() - start,
                      "rows": rows})


if __name__ == "__main__":
    main()
