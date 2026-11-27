#!/usr/bin/env python3
"""Extract inference-visible evidence features from sealed MuSiQue retrieval paths."""

import argparse
import gzip
import json
from pathlib import Path

import torch

from audit_kam_natural_data import digest, words
from run_kam_musique_recursive import BUDGETS
from run_kam_native_musique import INITIAL, corpus_and_fixture
from run_kam_natural_feasibility import Encoder

COLUMNS = ("dense_affinity", "initial_rank", "new_paragraph", "subquery_rank", "subquery_depth", "last_subquery",
           "read_for_binding", "query_body_overlap", "query_title_overlap", "query_mentions_title",
           "path_mentions_title", "mentions_path_title")


def packed(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def paragraph_features(question, doc, source, paths, affinity, length):
    q_set, q_text = question
    title, title_words, body, body_words = doc
    inverse, extra, branch, last, read = source
    return [round(affinity, 8), inverse if not extra else 0., float(extra), inverse if extra else 0.,
            branch / max(1, length - 1) if extra else 0., float(last), float(read),
            len(q_set & body_words) / max(1, len(q_set)), len(q_set & title_words) / max(1, len(q_set)),
            float(bool(title) and title in q_text),
            float(bool(title) and any(title in b for _, b in paths)),
            float(any(t and t in body for t, _ in paths))]


def build_case(learned, question, paragraphs, dots):
    count = learned["predicted_steps"]
    pool = learned["pool"]
    scope = list(dict.fromkeys(learned["read_from"]))
    paths = [(paragraphs[i][0], paragraphs[i][2]) for i in scope]
    source = [(1.0 / (rank + 1), False, 0, False, i in scope) for rank, i in enumerate(pool[:INITIAL])]
    if learned["fallback"]:
        source.extend((0., False, 0, False, False) for _ in pool[INITIAL:])
    else:
        boundary = INITIAL
        for branch, width in enumerate(BUDGETS[count]):
            source.extend((1.0 / (rank + 1), True, branch, branch + 1 == count, i in scope)
                          for rank, i in enumerate(pool[boundary:boundary+width]))
            boundary += width
    features = [paragraph_features(question, paragraphs[i], p, paths, value, count)
                for i, p, value in zip(pool, source, dots, strict=True)]
    return {"fallback": learned["fallback"], "pool": pool, "features": features}


@torch.inference_mode()
def run(a, articles, fixture):
    baseline = packed(a.baseline)
    learnt = [packed(file) for file in a.learned]
    versions = json.loads((a.assets / "revisions.json").read_text())
    fx, cx = digest(a.fixture), fixture["corpus_sha256"]
    assert baseline["models"] == versions
    assert all(p["baseline_sha256"] == digest(a.baseline) for p in learnt)
    assert all(p["fixture_sha256"] == fx and p["corpus_sha256"] == cx and (p["shard"], p["shards"]) == (a.shard, a.shards)
               for p in (baseline, *learnt))
    cases = fixture["rows"][a.shard::a.shards]
    active = {i for p in learnt for r in p["rows"] for i in r["pool"]}
    text = {}
    for i in active:
        doc = articles[i]
        t, b = words(doc["title"]), words(doc["body"])[:192]
        text[i] = (" " + " ".join(t) + " " if t else "", set(t), " " + " ".join(b) + " ", set(b))
    vs = sorted((torch.load(p, weights_only=True, map_location="cpu") for p in a.vectors), key=lambda r: r["begin"])
    assert all(x["corpus_sha256"] == cx and x["models"] == versions for x in vs)
    assert vs[0]["begin"] == 0 and vs[-1]["end"] == len(articles)
    assert all(l["end"] == r["begin"] for l, r in zip(vs, vs[1:]))
    bank = torch.cat([x["vectors"] for x in vs], 0).cuda()
    enc = Encoder(a.assets / "encoder", 128)
    vectors = enc([q["question"] for q in cases], "query")
    rows = []
    for j, (case, base, *fitted) in enumerate(zip(cases, baseline["rows"], *(r["rows"] for r in learnt), strict=True)):
        id_ = case["id"]
        assert base["id"] == id_ and all(q["id"] == id_ for q in fitted)
        s = words(case["question"])
        spec = set(s), " " + " ".join(s) + " "
        poses = torch.tensor([q["pool"] for q in fitted], device="cuda")
        dots = (bank[poses] @ vectors[j]).tolist()
        rows.append({"id": id_, "seeds": [build_case(q, spec, text, v) for q, v in zip(fitted, dots, strict=True)]})
    return {"columns": COLUMNS, "fixture_sha256": fx, "corpus_sha256": cx, "baseline_sha256": digest(a.baseline),
            "source_sha256": fixture["source_sha256"], "model_revisions": versions,
            "weights_sha256": [p["weights_sha256"] for p in learnt], "learned_sha256": [digest(p) for p in a.learned],
            "device": torch.cuda.get_device_name(), "shard": a.shard, "shards": a.shards, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "corpus", "fixture", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--learned", type=Path, nargs=3, required=True)
    p.add_argument("--vectors", type=Path, nargs=4, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    bank, fixture = corpus_and_fixture(a)
    record = run(a, bank, fixture)
    a.output.write_bytes(gzip.compress(json.dumps(record, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
