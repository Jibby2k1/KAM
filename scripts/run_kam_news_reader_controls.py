#!/usr/bin/env python3
"""Separate chunk visibility and reader limits on fixed MultiHop-RAG queries."""

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest, words
from run_kam_dataset_breadth import padded_article_chunks
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import measure_answer

MODES = ("original_pair", "context_pair", "sequential", "retrieved_gold_articles",
         "gold_evidence_text", "question_only")
SELECTIONS = {"original_pair": "one_pass_pair", "context_pair": "context_only",
              "sequential": "learned_sequential"}


def chunks(corpus):
    return [(row["title"], [text]) for row in corpus for text in padded_article_chunks(row["body"])]


def evidence_case(question, evidence):
    return {"question": question, "context": evidence}


def run(a):
    start = perf_counter()
    data, corpus = read(a.fixture), read(a.corpus)
    assert digest(a.corpus) == data["corpus_sha256"]
    archive = [read(path) for path in a.replays]
    assert all(r["fixture_sha256"] == digest(a.fixture) for r in archive)
    records = {row["id"]: row for shard in archive for row in shard["rows"]}
    assert len(records) == len(data["rows"])
    bank = chunks(corpus)
    article = {row["url"]: " ".join(words(row["body"])) for row in corpus}
    requests = {}
    outputs = []
    for case in data["rows"][a.shard::a.shards]:
        q = case["question"]
        replay = records[case["id"]]
        evidence = [(ev["title"], [ev["fact"]]) for ev in case["evidence"]]
        visibility = [" ".join(words(ev["fact"])) in article[ev["url"]] for ev in case["evidence"]]
        out = {"id": case["id"], "kind": case["kind"], "both_articles_retrieved": len(replay["gold"]) == 2,
               "both_fact_spans_retrieved": len(replay["visible_gold"]) == 2,
               "all_facts_in_source_articles": all(visibility), "controls": {}}
        cases = {name: evidence_case(q, [bank[replay["chunk_ids"][idx]]
                                        for idx in replay["arms"][arm]["pair"]])
                 for name, arm in SELECTIONS.items()}
        cases["gold_evidence_text"] = evidence_case(q, evidence)
        cases["question_only"] = evidence_case(q, [])
        if out["both_articles_retrieved"]:
            cases["retrieved_gold_articles"] = evidence_case(q, [bank[replay["chunk_ids"][idx]]
                                                                 for idx in replay["arms"]["oracle"]["pair"]])
        assert set(cases) <= set(MODES)
        requests[case["id"]] = (case["answer"], cases, out["controls"])
        outputs.append(out)

    models = {"small": a.assets / "reader", "base": a.base}
    for name, model in models.items():
        reader = Reader(model)
        prompts, details = [], []
        for truth, cases, out in requests.values():
            for mode, case in cases.items():
                prompts.append(reader.prompt(case, list(range(len(case["context"])))))
                details.append((truth, out.setdefault(mode, {})))
        unique = list(dict.fromkeys(prompts))
        predictions = dict(zip(unique, reader.answers(unique), strict=True))
        for prompt, (answer, target) in zip(prompts, details, strict=True):
            p = predictions[prompt]
            target[name] = {**measure_answer(p, answer),
                            "token_overlap": bool(set(p.casefold().split()) & set(answer.casefold().split()))}
        del reader

    scores = {}
    for mode in MODES:
        scores[mode] = {}
        for model in models:
            arm = [r["controls"][mode][model] for r in outputs if mode in r["controls"]]
            scores[mode][model] = {k: sum(r[k] for r in arm) for k in ("em", "token_overlap")} | {"n": len(arm)}
    save(a.output, {"fixture_sha256": digest(a.fixture), "base_files_sha256": digest(a.base / "files.sha256"),
                    "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
                    "scores": scores, "kind_counts": dict(Counter(r["kind"] for r in outputs)),
                    "elapsed_seconds": perf_counter() - start, "rows": outputs})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "base", "fixture", "corpus", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--replays", type=Path, nargs="+", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    run(a)


if __name__ == "__main__":
    main()
