#!/usr/bin/env python3
"""Compare matched-width paragraph expansion and a decomposition oracle."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from zipfile import ZipFile

from audit_kam_natural_data import digest
from run_kam_native_musique import DEPTH, INITIAL, corpus_and_fixture

REFERENCE = re.compile(r"#(\d+)\b")
BUDGETS = {2: (32, 32), 3: (22, 21, 21), 4: (16, 16, 16, 16)}
ARMS = ("parallel_three", "recursive_three", "annotated_only", "resolved_answers")


def freeze_oracle(args, articles, fixture):
    from prepare_kam_native_musique import MEMBERS, canonical_paragraph

    source = args.source
    assert digest(source) == fixture["source_sha256"]
    required = {q["id"]: q for q in fixture["rows"]}
    paragraph_ids = {canonical_paragraph({"title": p["title"], "paragraph_text": p["body"]}): p["id"]
                     for p in articles}
    assert len(paragraph_ids) == len(articles)
    oracle = {}
    member_hash = sha256()
    with ZipFile(source) as archive, archive.open(MEMBERS["dev"]) as raw:
        for line in raw:
            member_hash.update(line)
            q = json.loads(line)
            if q["id"] not in required:
                continue
            frozen = required[q["id"]]
            assert frozen["question"] == q["question"]
            decomp = q["question_decomposition"]
            assert len(decomp) == frozen["hops"]
            annotated, resolved, labels, answers = [], [], [], []
            for step in decomp:
                question, answer = step["question"], step["answer"]
                assert all(0 < int(ref) <= len(answers) for ref in REFERENCE.findall(question))
                annotated.append(REFERENCE.sub("", question).strip())
                resolved.append(REFERENCE.sub(lambda match: answers[int(match.group(1)) - 1], question))
                target = q["paragraphs"][step["paragraph_support_idx"]]
                assert target["is_supporting"]
                labels.append(paragraph_ids[canonical_paragraph(target)])
                answers.append(answer)
            assert sorted(labels) == frozen["gold"]
            oracle[q["id"]] = {"annotated_only": annotated, "resolved_answers": resolved, "gold_in_order": labels}
    assert member_hash.hexdigest() == fixture["member_sha256"]["dev"]
    assert oracle.keys() == required.keys()
    args.output.write_text(json.dumps({"fixture_sha256": digest(args.fixture), "rows": oracle},
                                     separators=(",", ":")) + "\n")


def ranked_additions(rank, pool, budget):
    present = set(pool)
    candidates = [item for item in rank if item not in present][:budget]
    assert len(candidates) == len(set(candidates)) == budget
    return candidates


def search(args, articles, fixture):
    import torch

    from run_kam_natural_feasibility import Encoder

    torch.set_float32_matmul_precision("highest")
    started = perf_counter()
    models = json.loads((args.assets / "revisions.json").read_text())
    vectors = sorted((torch.load(x, weights_only=True, map_location="cpu") for x in args.vectors),
                     key=lambda x: x["begin"])
    corpus_hash = fixture["corpus_sha256"]
    fixture_hash = digest(args.fixture)
    assert all(v["corpus_sha256"] == corpus_hash and v["models"] == models for v in vectors)
    assert vectors[0]["begin"] == 0 and vectors[-1]["end"] == len(articles)
    assert all(a["end"] == b["begin"] for a, b in zip(vectors, vectors[1:]))
    bank = torch.cat([v["vectors"] for v in vectors], 0).cuda()
    baseline = json.loads(gzip.decompress(args.baseline.read_bytes()))
    assert (baseline["fixture_sha256"], baseline["corpus_sha256"], baseline["models"]) == (
        fixture_hash, corpus_hash, models)
    assert (baseline["shard"], baseline["shards"]) == (args.shard, args.shards)
    original = {q["id"]: q["dense"][:INITIAL] for q in baseline["rows"]}
    published_single = {q["id"]: q["one_anchor"][INITIAL:] for q in baseline["rows"]}
    cases = fixture["rows"][args.shard::args.shards]
    assert list(original) == [q["id"] for q in cases]
    authorized = json.loads(args.oracle.read_text())
    assert authorized["fixture_sha256"] == fixture_hash
    encoder = Encoder(args.assets / "encoder", 128)
    output = {}
    for i in range(0, len(cases), 16):
        group = cases[i:i+16]
        questions = [q["question"] for q in group]
        initial = [original[q["id"]] for q in group]
        query_vec = encoder(questions, "query")

        def prompt(question, path):
            if len(path) == 1:
                article = articles[path[0]]
                return f"{question} First retrieved document: {article['title']}. {article['body']}"
            # Bound each prefix so later anchors can reach the token cap.
            context = "".join(f" Retrieved document: {articles[d]['title']}. "
                              + " ".join(articles[d]["body"].split()[:24]) for d in path)
            return question + context

        def ranks(branches):
            count = len(branches[0])
            prompts = [prompt(q, path) for q, paths in zip(questions, branches, strict=True) for path in paths]
            embedded = encoder(prompts, "query").reshape(len(questions), count, -1)
            updated = (embedded + query_vec[:, None]) * 0.5
            return (updated @ bank.T).argsort(dim=-1, descending=True, stable=True)[:, :, :DEPTH].tolist()

        first = [published_single[q["id"]] for q in group]
        assert all(len(r) == len(set(r)) == INITIAL and set(r).isdisjoint(p)
                   for r, p in zip(first, initial, strict=True))
        fixed = ranks([[[p[1]], [p[2]]] for p in initial])
        for mode in ARMS[:2]:
            pools = [p.copy() for p in initial]
            paths = [[p[0]] for p in initial]
            for step, budget in enumerate((22, 21, 21)):
                if step == 0:
                    candidate_ranks = first
                elif mode == "parallel_three":
                    candidate_ranks = [columns[step - 1] for columns in fixed]
                else:
                    candidate_ranks = [columns[0] for columns in ranks([[p] for p in paths])]
                for pool, path, ranked in zip(pools, paths, candidate_ranks, strict=True):
                    new = ranked_additions(ranked, pool, budget)
                    pool.extend(new)
                    if mode == "recursive_three":
                        path.append(new[0])
            for q, p in zip(group, pools, strict=True):
                output.setdefault(q["id"], {"id": q["id"]})[mode] = p

        for mode in ARMS[2:]:
            # Oracle subquestions receive no original-question embedding.
            inputs = [text for q in group for text in authorized["rows"][q["id"]][mode]]
            all_ranks = (encoder(inputs, "query") @ bank.T).argsort(dim=-1, descending=True, stable=True)
            all_ranks = iter(all_ranks[:, :DEPTH].tolist())
            for q, starter in zip(group, initial, strict=True):
                labels = authorized["rows"][q["id"]]["gold_in_order"]
                assert sorted(labels) == q["gold"]
                pool = starter.copy()
                diagnostic = []
                for budget, target in zip(BUDGETS[q["hops"]], labels, strict=True):
                    ranked = next(all_ranks)
                    diagnostic.append(ranked.index(target) + 1 if target in ranked else None)
                    pool.extend(ranked_additions(ranked, pool, budget))
                output[q["id"]][mode] = pool
                output[q["id"]][mode + "_target_ranks"] = diagnostic
            assert next(all_ranks, None) is None
    result = {"device": torch.cuda.get_device_name(), "models": models, "fixture_sha256": fixture_hash,
              "corpus_sha256": corpus_hash, "oracle_sha256": digest(args.oracle),
              "baseline_sha256": digest(args.baseline), "shard": args.shard, "shards": args.shards,
              "elapsed_seconds": perf_counter() - started, "rows": list(output.values())}
    args.output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "search"))
    for name in ("corpus", "fixture", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source", "assets", "baseline", "oracle"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--vectors", type=Path, nargs="+")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    docs, fixture = corpus_and_fixture(a)
    (freeze_oracle if a.mode == "freeze" else search)(a, docs, fixture)


if __name__ == "__main__":
    main()
