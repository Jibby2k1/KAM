#!/usr/bin/env python3
"""Extract fixed E5 and visible hyperlink features for supervised IIRC routing."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from audit_kam_natural_data import digest, words
from run_kam_hotpot_pair_scoring import links
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts
from run_kam_natural_router import score

DOCUMENT_COLUMNS = ("plain_query_article", "source_query_article", "question_link_excerpt",
                    "source_query_link_excerpt", "question_article_overlap", "question_link_overlap")
PAIR_COLUMNS = ("article_cosine", "excerpt_cosine", "question_complement")
TITLE_COLUMNS = ("one_way_or_more", "both_directions")


def corpus_vectors(texts, encoder, limit=192):
    unique = list(dict.fromkeys(v for group in texts for v in group))
    result = encoder(unique, "passage", max_length=limit)
    return dict(zip(unique, result, strict=True))


@torch.inference_mode()
def evaluate(rows, encoder, adapter):
    cases = [{"context": [(t, [body]) for t, body in row["candidates"]]} for row in rows]
    banks = [candidate_texts(case) for case in cases]
    anchors = [[f"{title}. {excerpt}" for (title, _), excerpt in zip(row["candidates"], row["link_context"], strict=True)]
               for row in rows]
    doc_vectors = corpus_vectors(banks, encoder)
    anchor_vectors = corpus_vectors(anchors, encoder)
    plain = adapter(encoder([row["question"] for row in rows], "query", max_length=256))
    contextual = adapter(encoder([f'{row["question"]} Source paragraph: {row["source_excerpt"]}'
                                  for row in rows], "query", max_length=256))
    output = []
    for i, row in enumerate(rows):
        docs = torch.stack([doc_vectors[t] for t in banks[i]])
        local = torch.stack([anchor_vectors[t] for t in anchors[i]])
        doc_cos = docs @ docs.T
        anchor_cos = local @ local.T
        n = len(docs)
        idx = torch.triu_indices(n, n, 1, device=docs.device)
        left, right = idx
        mentions = links([cases[i]])[0].to(docs.device)
        a, b = mentions[left, right], mentions[right, left]
        q_terms = set(words(row["question"]))
        normalizer = max(1, len(q_terms))
        doc_terms = [q_terms.intersection(words(text)) for text in banks[i]]
        local_terms = [q_terms.intersection(words(text)) for text in anchors[i]]
        lexical = torch.tensor([[len(t) / normalizer, len(v) / normalizer]
                                for t, v in zip(doc_terms, local_terms, strict=True)], device=docs.device)
        query_columns = [score(q[i:i+1], vecs.unsqueeze(0))[0]
                         for vecs in (docs, local) for q in (plain, contextual)]
        doc_data = torch.column_stack((*query_columns, lexical))
        complement = torch.tensor([(len(doc_terms[a] | doc_terms[b]) - max(len(doc_terms[a]), len(doc_terms[b]))) /
                                   normalizer for a, b in idx.T.tolist()], device=docs.device)
        pair_data = torch.column_stack((doc_cos[left, right], anchor_cos[left, right], complement, a | b, a & b))
        pool = sha256(json.dumps(row["candidates"], ensure_ascii=False).encode()).hexdigest()
        output.append({"id": row["id"], "source_id": row["source_id"], "pool_sha256": pool,
                       "gold": row["gold_indices"], "docs": doc_data.tolist(), "pairs": pair_data.tolist()})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "assets", "config", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    fixture = json.loads(a.fixture.read_text())
    cfg = json.loads(a.config.read_text())
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    encoder = Encoder(a.assets / "encoder", 128)
    rows = evaluate(fixture["rows"][a.shard::a.shards], encoder, adapter)
    payload = {"fixture_sha256": digest(a.fixture), "source_sha256": fixture["source_sha256"],
               "config_sha256": digest(a.config), "baseline_sha256": a.baseline_sha256,
               "assets": json.loads((a.assets / "revisions.json").read_text()), "device": torch.cuda.get_device_name(),
               "torch": torch.__version__, "shard": a.shard, "shards": a.shards,
               "doc_columns": DOCUMENT_COLUMNS, "pair_columns": PAIR_COLUMNS + TITLE_COLUMNS, "rows": rows}
    a.output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
