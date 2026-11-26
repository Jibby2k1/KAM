#!/usr/bin/env python3
"""Compare one, two, and four conditioned anchors at an equal candidate budget."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_conditioned_corpus_search import load_gzip, top_corpus
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_hotpot_independent import read
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts
from run_kam_natural_router import score
from run_kam_sentence_verifier_fullwiki import pool_text

ADDITIONS = 64
ANCHORS = 4
BATCH = 16


def balanced_additions(branches, original):
    """Interleave ranks, skip repeated or already visible articles, and stop at the budget."""
    seen = set(original)
    added = []
    for ranks in zip(*branches, strict=True):
        for doc in ranks:
            if doc not in seen:
                seen.add(doc)
                added.append(doc)
                if len(added) == ADDITIONS:
                    return added
    raise ValueError("Insufficient distinct additions")


@torch.inference_mode()
def run(a):
    t0 = perf_counter()
    cfg, fixture, manifest = (read(p) for p in (a.config, a.fixture, a.inputs / "manifest.json"))
    sealed = load_gzip(a.pools)
    index_sha = digest(a.inputs / "lexical" / "completed.json")
    assert sealed["fixture_sha256"] == digest(a.fixture) and sealed["index_sha256"] == index_sha
    assert fixture["source_sha256"] == manifest["source_sha256"]
    cases = fixture["rows"]["test"]
    assert len(cases) == len(sealed["rows"]) == 1024 and set(sealed["rows"]) == {q["id"] for q in cases}
    cases = cases[a.shard::a.shards]
    index = FullWikiIndex(a.inputs, manifest, cfg)
    encoder = Encoder(a.assets / "encoder", 128)
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    rows = []
    for offset in range(0, len(cases), BATCH):
        chunk = cases[offset:offset+BATCH]
        pools = [sealed["rows"][q["id"]] for q in chunk]
        examples = [pool_text(index.db, q["question"], p) for q, p in zip(chunk, pools, strict=True)]
        adapted = adapter(encoder([q["question"] for q in chunk], "query"))
        ids = torch.tensor(pools, device="cuda")
        slots = score(adapted, index.bank[ids]).argsort(dim=1, descending=True, stable=True)[:, :ANCHORS]
        anchor_ids = ids.gather(1, slots).tolist()
        texts = []
        for example, selected in zip(examples, slots.tolist(), strict=True):
            passages = candidate_texts(example)
            texts.extend(f"{example['question']} First retrieved document: {passages[i]}" for i in selected)
        conditioned = adapter(encoder(texts, "query")).reshape(len(chunk), ANCHORS, -1)
        original = adapted.unsqueeze(1)
        queries = torch.cat((original, (original + conditioned) * .5), dim=1).flatten(0, 1)
        # Original-pool IDs are removed after search. Depth 128 leaves at least 64 additions
        # even if all original articles would otherwise be returned.
        excluded = [selections[0] for selections in anchor_ids for _ in range(ANCHORS + 1)]
        hits, _ = top_corpus(index, queries, excluded, depth=128)
        for j, (case, pool, anchors) in enumerate(zip(chunk, pools, anchor_ids, strict=True)):
            start = j * (ANCHORS + 1)
            control, *expanded = hits[start:start + ANCHORS + 1]
            lists = {"unconditioned": balanced_additions([control], pool)}
            lists.update({f"anchors_{n}": balanced_additions(expanded[:n], pool) for n in (1, 2, 4)})
            rows.append({"id": case["id"], "anchors": anchors, "added": lists,
                         "pool_hash": sha256(json.dumps(pool, separators=(",", ":")).encode()).hexdigest()})
    return {"fixture_sha256": digest(a.fixture), "pools_sha256": digest(a.pools), "config_sha256": digest(a.config),
            "index_sha256": index_sha, "baseline_sha256": a.baseline_sha256,
            "models": read(a.assets / "revisions.json"), "gpu": torch.cuda.get_device_name(),
            "indexed_documents": index.meta["indexed"], "shard": a.shard, "shards": a.shards,
            "seconds": perf_counter() - t0, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "inputs", "config", "fixture", "pools", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    data = json.dumps(run(a), separators=(",", ":"), allow_nan=False).encode()
    a.output.write_bytes(gzip.compress(data, mtime=0))


if __name__ == "__main__":
    main()
