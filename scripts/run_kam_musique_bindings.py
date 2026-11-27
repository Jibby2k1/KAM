#!/usr/bin/env python3
"""Read intermediate answers under fixed author-supplied MuSiQue templates."""

import argparse
import gzip
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

from audit_kam_natural_data import digest
from run_kam_musique_recursive import BUDGETS, REFERENCE, ranked_additions
from run_kam_native_musique import DEPTH, INITIAL, corpus_and_fixture

ARMS = ("top1", "top3", "gold_paragraph")
BASE_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
BASE_WEIGHT = "1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8"


def freeze(a, articles, fixture):
    from prepare_kam_native_musique import MEMBERS, canonical_paragraph

    previous_oracle = json.loads(a.oracle.read_text())
    assert digest(a.fixture) == previous_oracle["fixture_sha256"]
    assert digest(a.source) == fixture["source_sha256"]
    by_paragraph = {canonical_paragraph({"title": r["title"], "paragraph_text": r["body"]}): r["id"] for r in articles}
    qids = {q["id"]: q["gold"] for q in fixture["rows"]}
    pure, labels = {}, {}
    member_digest = sha256()
    with ZipFile(a.source) as archive, archive.open(MEMBERS["dev"]) as source:
        for line in source:
            member_digest.update(line)
            q = json.loads(line)
            id_ = q["id"]
            if id_ not in qids:
                continue
            templates, answers, targets = [], [], []
            for step, hop in enumerate(q["question_decomposition"], 1):
                template = hop["question"]
                assert all(1 <= int(i) < step for i in REFERENCE.findall(template))
                templates.append(template)
                answers.append(hop["answer"])
                gold = q["paragraphs"][hop["paragraph_support_idx"]]
                assert gold["is_supporting"]
                targets.append(by_paragraph[canonical_paragraph(gold)])
            checked = previous_oracle["rows"][id_]
            assert [REFERENCE.sub("", s).strip() for s in templates] == checked["annotated_only"]
            assert [REFERENCE.sub(lambda m: answers[int(m.group(1)) - 1], s) for s in templates] == checked["resolved_answers"]
            assert targets == checked["gold_in_order"] and sorted(targets) == qids[id_]
            pure[id_] = templates
            labels[id_] = {"targets": targets, "answers": answers}
    assert member_digest.hexdigest() == fixture["member_sha256"]["dev"] and pure.keys() == labels.keys() == qids.keys()
    a.output.write_text(json.dumps({"fixture_sha256": digest(a.fixture), "templates": pure}, separators=(",", ":")) + "\n")
    a.labels.write_text(json.dumps({"fixture_sha256": digest(a.fixture), "templates_sha256": digest(a.output),
                                    "rows": labels}, separators=(",", ":")) + "\n")


def binding_queries(templates, previous, step):
    return [REFERENCE.sub(lambda match: bindings[int(match.group(1)) - 1], row[step]).strip()
            for row, bindings in zip(templates, previous, strict=True)]


def score_case(rows, generated, source, target):
    from run_kam_natural_feasibility import measure_answer

    for row, reply, actual, label in zip(rows, generated, source, target, strict=True):
        row["answer_em"].append(measure_answer(reply, actual)["em"] if reply else 0)
        row["gold_in_reader"].append(label in row["reader_context_ids"][-1])


def replay(arm, current, templates, labels, initial, passage_text, bank, encoder, reader, *,
           gold_at=None, alternate_reader=None, alternate_steps=()):
    pools = [initial[q["id"]].copy() for q in current]
    rows = [{"id": q["id"], "target_ranks": [], "answer_em": [], "reader_context_ids": [],
             "gold_in_reader": [], "unfilled_references": 0} for q in current]
    histories = [[] for _ in current]
    for step, add in enumerate(BUDGETS[current[0]["hops"]]):
        queries = binding_queries(templates, histories, step)
        assert all(q for q in queries)
        updated = encoder(queries, "query")
        ranks = (updated @ bank.T).argsort(dim=-1, descending=True, stable=True)[:, :DEPTH].tolist()
        raw_targets = [label["targets"][step] for label in labels]
        selected = []
        for pool, row, rank, target, template, previous in zip(pools, rows, ranks, raw_targets, templates, histories, strict=True):
            pool.extend(ranked_additions(rank, pool, add))
            row["target_ranks"].append(rank.index(target) + 1 if target in rank else None)
            row["unfilled_references"] += sum(previous[int(i)-1] == "unknown" for i in REFERENCE.findall(template[step]))
            # Only this labeled diagnostic gives the reader the target paragraph.
            evidence = [target] if arm == "gold_paragraph" else rank[:1 if arm == "top1" else 3]
            selected.append(evidence)
            row["reader_context_ids"].append(evidence)
        prompts = [f"Answer the question with a short answer. Question: {question} Evidence: "
                   + " ".join(passage_text(i) for i in ids)
                   for question, ids in zip(queries, selected, strict=True)]
        source_reader = alternate_reader if step in alternate_steps else reader
        predictions = source_reader.answers(prompts)
        score_case(rows, predictions, [label["answers"][step] for label in labels], raw_targets)
        for previous, answer, label in zip(histories, predictions, labels, strict=True):
            binding = label["answers"][step] if step == gold_at else answer
            previous.append(" ".join(binding.split()) or "unknown")
    for row, pool in zip(rows, pools, strict=True):
        row["pool"] = pool
        assert len(pool) == len(set(pool)) == DEPTH
    return rows


def run(a, articles, fixture):
    import torch

    from run_kam_hotpot_pool_scaling import Reader
    from run_kam_natural_feasibility import Encoder

    torch.set_float32_matmul_precision("highest")
    start = perf_counter()
    fx, cx = digest(a.fixture), fixture["corpus_sha256"]
    models = json.loads((a.assets / "revisions.json").read_text())
    blocks = sorted((torch.load(p, weights_only=True, map_location="cpu") for p in a.vectors), key=lambda x: x["begin"])
    assert all(v["corpus_sha256"] == cx and v["models"] == models for v in blocks)
    assert blocks[0]["begin"] == 0 and blocks[-1]["end"] == len(articles)
    assert all(x["end"] == y["begin"] for x, y in zip(blocks, blocks[1:]))
    bank = torch.cat([v["vectors"] for v in blocks], 0).cuda()
    prior = json.loads(gzip.decompress(a.baseline.read_bytes()))
    assert prior["corpus_sha256"] == cx and prior["fixture_sha256"] == fx and prior["models"] == models
    assert (prior["shard"], prior["shards"]) == (a.shard, a.shards)
    initial = {r["id"]: r["dense"][:INITIAL] for r in prior["rows"]}
    source_templates = json.loads(a.templates.read_text())
    analysis_labels = json.loads(a.labels.read_text())
    assert source_templates["fixture_sha256"] == analysis_labels["fixture_sha256"] == fx
    assert analysis_labels["templates_sha256"] == digest(a.templates)
    encoder = Encoder(a.assets / "encoder", 128)
    base_path = a.assets / f"reader_base_{BASE_REVISION}"
    assert digest(base_path / "model.safetensors") == BASE_WEIGHT
    reader = Reader(base_path)
    larger = None
    if a.reader_size:
        from kam_flan_reader_assets import LARGE_REVISION, LARGE_WEIGHT

        large_path = a.assets / f"reader_large_{LARGE_REVISION}"
        assert digest(large_path / "model.safetensors") == LARGE_WEIGHT
        larger = Reader(large_path)
        assert larger.tokenizer.get_vocab() == reader.tokenizer.get_vocab()
        models = models | {"google/flan-t5-large": LARGE_REVISION}

    @lru_cache(maxsize=8192)
    def passage_text(i):
        passage = articles[i]
        data = reader.tokenizer(passage["title"] + ". " + passage["body"], add_special_tokens=False,
                                truncation=True, max_length=128)
        return reader.tokenizer.decode(data.input_ids)

    questions = fixture["rows"][a.shard::a.shards]
    assert list(initial) == [q["id"] for q in questions]
    answers = {}
    for nhops in BUDGETS:
        cases = [q for q in questions if q["hops"] == nhops]
        if not cases:
            continue
        templates = [source_templates["templates"][q["id"]] for q in cases]
        diagnostics = [analysis_labels["rows"][q["id"]] for q in cases]
        assert all(len(p) == nhops for p in templates)
        policies = [(arm, arm, None, ()) for arm in ARMS]
        if a.single_oracle:
            policies = [("top1", "top1", None, ()),
                        *((f"gold_at_{i+1}", "top1", i, ()) for i in range(nhops - 1))]
        if a.reader_size:
            policies = [("top1", "top1", None, ()), ("large_first", "top1", None, (0,))]
            if nhops > 2:
                policies.extend([("large_penultimate", "top1", None, (nhops - 2,)),
                                 ("large_all_intermediate", "top1", None, tuple(range(nhops - 1)))])
            policies.append(("large_final", "top1", None, (nhops - 1,)))
        for arm, context_rule, gold_at, alternate_steps in policies:
            for row in replay(context_rule, cases, templates, diagnostics, initial, passage_text, bank, encoder, reader,
                              gold_at=gold_at, alternate_reader=larger, alternate_steps=alternate_steps):
                answers.setdefault(row["id"], {})[arm] = row
    return {"fixture_sha256": fx, "corpus_sha256": cx, "models": models | {"google/flan-t5-base": BASE_REVISION},
            "templates_sha256": digest(a.templates), "scoring_labels_sha256": digest(a.labels),
            "baseline_sha256": digest(a.baseline), "elapsed_seconds": perf_counter() - start,
            "shard": a.shard, "shards": a.shards, "device": torch.cuda.get_device_name(),
            "rows": [{"id": q["id"], "arms": answers[q["id"]]} for q in questions]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "run"))
    for name in ("corpus", "fixture", "output", "labels"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source", "oracle", "assets", "baseline", "templates"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--vectors", nargs=4, type=Path)
    diagnostics = p.add_mutually_exclusive_group()
    diagnostics.add_argument("--single-oracle", action="store_true")
    diagnostics.add_argument("--reader-size", action="store_true")
    p.add_argument("--shard", default=0, type=int)
    p.add_argument("--shards", default=1, type=int)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    articles, fixture = corpus_and_fixture(a)
    if a.mode == "freeze":
        freeze(a, articles, fixture)
    else:
        out = run(a, articles, fixture)
        a.output.write_bytes(gzip.compress(json.dumps(out, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
