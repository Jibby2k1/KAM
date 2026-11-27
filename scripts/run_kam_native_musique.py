#!/usr/bin/env python3
"""Build and search the exact native MuSiQue paragraph corpus."""

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

from audit_kam_natural_data import digest, words

DEPTH = 128
INITIAL = 64
COMPONENT_DEPTH = 256


def corpus_and_fixture(args):
    cfg = json.loads(args.fixture.read_text())
    assert digest(args.corpus) == cfg["corpus_sha256"]
    with gzip.open(args.corpus, "rt") as stream:
        articles = [json.loads(line) for line in stream]
    assert [r["id"] for r in articles] == list(range(cfg["documents"]))
    return articles, cfg


def lexical(args, articles, fixture):
    import tantivy

    args.output.mkdir(exist_ok=False)
    index_dir = args.output / "tantivy"
    index_dir.mkdir()
    builder = tantivy.SchemaBuilder()
    builder.add_integer_field("row_id", stored=True)
    builder.add_text_field("contents", index_option="freq")
    schema = builder.build()
    index = tantivy.Index(schema, path=str(index_dir))
    writer = index.writer(heap_size=256 * 1024 * 1024, num_threads=4)
    for article in articles:
        writer.add_document(tantivy.Document.from_dict({"row_id": article["id"],
                                                       "contents": article["title"] + ". " + article["body"]}, schema))
    writer.commit()
    writer.wait_merging_threads()
    index.reload()
    assert index.searcher().num_docs == fixture["documents"]
    meta = {"corpus_sha256": fixture["corpus_sha256"], "documents": fixture["documents"],
            "tantivy": tantivy.__version__}
    (args.output / "meta.json").write_text(json.dumps(meta))


def embed(args, articles, fixture):
    import torch

    from run_kam_natural_feasibility import Encoder

    torch.set_float32_matmul_precision("highest")
    n = len(articles)
    a, b = (n * j // args.shards for j in (args.shard, args.shard + 1))
    enc = Encoder(args.assets / "encoder", 160)
    vectors = enc([a["title"] + ". " + a["body"] for a in articles[a:b]], "passage").cpu()
    assert vectors.dtype == torch.float32
    torch.save({"begin": a, "end": b, "vectors": vectors, "corpus_sha256": fixture["corpus_sha256"],
                "models": json.loads((args.assets / "revisions.json").read_text()),
                "gpu": torch.cuda.get_device_name()}, args.output)


def lexical_rank(index, searcher, question):
    terms = " ".join(f'"{text}"' for text in sorted(set(words(question))))
    query = index.parse_query(terms, ["contents"])
    hits = searcher.search(query, limit=COMPONENT_DEPTH, count=False).hits
    ids = [searcher.doc(address).get_first("row_id") for _, address in hits]
    if len(ids) < COMPONENT_DEPTH:
        used = set(ids)
        ids.extend(i for i in range(COMPONENT_DEPTH) if i not in used)
        ids = ids[:COMPONENT_DEPTH]
    assert len(ids) == len(set(ids)) == COMPONENT_DEPTH
    return ids


def search(args, articles, fixture):
    import tantivy
    import torch

    from run_kam_fullwiki_native_compare import reciprocal_rank_fusion
    from run_kam_multi_anchor_expansion import balanced_additions
    from run_kam_natural_feasibility import Encoder

    started = perf_counter()
    torch.set_float32_matmul_precision("highest")
    models = json.loads((args.assets / "revisions.json").read_text())
    vectors = sorted((torch.load(x, weights_only=True, map_location="cpu") for x in args.vectors),
                     key=lambda x: x["begin"])
    assert all(v["corpus_sha256"] == fixture["corpus_sha256"] and v["models"] == models for v in vectors)
    assert vectors[0]["begin"] == 0 and vectors[-1]["end"] == len(articles)
    assert all(a["end"] == b["begin"] for a, b in zip(vectors, vectors[1:]))
    bank = torch.cat([v["vectors"] for v in vectors], 0).cuda()
    assert len(bank) == len(articles)
    lexical_meta = json.loads((args.lexical / "meta.json").read_text())
    assert lexical_meta["corpus_sha256"] == fixture["corpus_sha256"] and lexical_meta["documents"] == len(articles)
    index = tantivy.Index.open(str(args.lexical / "tantivy"))
    searcher = index.searcher()
    assert searcher.num_docs == len(articles)
    encoder = Encoder(args.assets / "encoder", 128)
    cases = fixture["rows"][args.shard::args.shards]
    output = []
    for i in range(0, len(cases), 16):
        chunk = cases[i:i+16]
        p = [q["question"] for q in chunk]
        query = encoder(p, "query")
        semantic = (query @ bank.T).argsort(dim=-1, descending=True, stable=True)[:, :COMPONENT_DEPTH].tolist()
        texts = []
        for question, ranked in zip(p, semantic, strict=True):
            for a in ranked[:2]:
                doc = articles[a]
                texts.append(f"{question} First retrieved document: {doc['title']}. {doc['body']}")
        condition = encoder(texts, "query").reshape(len(chunk), 2, -1)
        updated = (condition + query[:, None]) * 0.5
        extra = (updated @ bank.T).argsort(dim=-1, descending=True, stable=True)[:, :, :DEPTH].tolist()
        for q, raw, fresh in zip(chunk, semantic, extra, strict=True):
            lexical_ids = lexical_rank(index, searcher, q["question"])
            hybrid = reciprocal_rank_fusion(lexical_ids, raw, DEPTH, 60)
            original = raw[:INITIAL]
            single = balanced_additions(fresh[:1], original)
            wide = balanced_additions(fresh, original)
            assert len(single) == len(wide) == INITIAL
            output.append({"id": q["id"], "dense": raw[:DEPTH], "bm25": lexical_ids[:DEPTH], "hybrid": hybrid,
                           "one_anchor": original + single, "two_anchors": original + wide})
    result = {"device": torch.cuda.get_device_name(), "fixture_sha256": digest(args.fixture),
              "corpus_sha256": fixture["corpus_sha256"], "models": models, "shard": args.shard,
              "shards": args.shards, "elapsed_seconds": perf_counter() - started, "rows": output}
    args.output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("lexical", "embed", "search"))
    for name in ("corpus", "fixture", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--lexical", type=Path)
    p.add_argument("--vectors", type=Path, nargs="+")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    docs, fixture = corpus_and_fixture(a)
    {"lexical": lexical, "embed": embed, "search": search}[a.mode](a, docs, fixture)


if __name__ == "__main__":
    main()
