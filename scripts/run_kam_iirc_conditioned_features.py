#!/usr/bin/env python3
"""Extract second-hop E5 features for fixed IIRC anchor choices."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from audit_kam_natural_data import digest
from run_kam_iirc_pair_features import corpus_vectors
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts
from run_kam_natural_router import score


@torch.inference_mode()
def evaluate(rows, first, encoder, adapter):
    docs = [candidate_texts({"context": [(t, [b]) for t, b in r["candidates"]]}) for r in rows]
    local = [[f"{title}. {excerpt}" for (title, _), excerpt in zip(r["candidates"], r["link_context"], strict=True)]
             for r in rows]
    vectors = corpus_vectors(docs, encoder)
    highlights = corpus_vectors(local, encoder)
    choices, prompts = [], []
    for r, bank in zip(rows, docs, strict=True):
        anchors = sorted(set(r["gold_indices"]) | ({first[r["id"]]} if first is not None else set()))
        source = f'{r["question"]} Source paragraph: {r["source_excerpt"]}'
        for anchor in anchors:
            excerpt = " ".join(bank[anchor].split()[:48])
            prompts.append(f"{source} First retrieved document: {excerpt}")
        choices.append(anchors)
    vectors2 = adapter(encoder(prompts, "query", max_length=256))
    output, offset = [], 0
    for r, bank, snippets, anchors in zip(rows, docs, local, choices, strict=True):
        article_keys = torch.stack([vectors[t] for t in bank]).unsqueeze(0)
        local_keys = torch.stack([highlights[t] for t in snippets]).unsqueeze(0)
        results = {}
        for x in anchors:
            q = vectors2[offset:offset + 1]
            results[str(x)] = torch.column_stack((score(q, article_keys)[0], score(q, local_keys)[0])).tolist()
            offset += 1
        pool = sha256(json.dumps(r["candidates"], ensure_ascii=False).encode()).hexdigest()
        output.append({"id": r["id"], "source_id": r["source_id"], "pool_sha256": pool,
                       "conditioned": results})
    assert offset == len(prompts)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "assets", "config", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--first", type=Path)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    fixture = json.loads(a.fixture.read_text())
    first = {r["id"]: r["first"] for r in json.loads(a.first.read_text())["rows"]} if a.first else None
    assert len(fixture["rows"]) == (153 if first is not None else 1200)
    if first is not None:
        assert first.keys() == {r["id"] for r in fixture["rows"]}
    adapter, _ = restore_checkpoint(a.baseline, a.config, json.loads(a.config.read_text()), None, None,
                                    weights_sha256=a.baseline_sha256)
    encoder = Encoder(a.assets / "encoder", 128)
    records = evaluate(fixture["rows"][a.shard::a.shards], first, encoder, adapter)
    payload = {"fixture_sha256": digest(a.fixture), "source_sha256": fixture["source_sha256"],
               "config_sha256": digest(a.config), "baseline_sha256": a.baseline_sha256,
               "assets": json.loads((a.assets / "revisions.json").read_text()), "gpu": torch.cuda.get_device_name(),
               "torch": torch.__version__, "shard": a.shard, "shards": a.shards,
               "first_sha256": digest(a.first) if a.first else None, "rows": records}
    a.output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
