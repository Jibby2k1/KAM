#!/usr/bin/env python3
"""Compare original-question reading at a fixed four-paragraph budget."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

from audit_kam_natural_data import digest
from run_kam_musique_recursive import BUDGETS
from run_kam_native_musique import corpus_and_fixture

LIMIT = 4
PASSAGE_CAP = 96


def freeze(a, fixture):
    from prepare_kam_native_musique import MEMBERS

    assert digest(a.source) == fixture["source_sha256"]
    cases = {q["id"]: q for q in fixture["rows"]}
    answers = {}
    hasher = sha256()
    with ZipFile(a.source) as archive, archive.open(MEMBERS["dev"]) as source:
        for line in source:
            hasher.update(line)
            row = json.loads(line)
            i = row["id"]
            if i in cases:
                assert row["question"] == cases[i]["question"] and row["answerable"]
                answers[i] = list(dict.fromkeys([row["answer"], *row["answer_aliases"]]))
                assert all(type(a) is str and a.strip() for a in answers[i])
    assert hasher.hexdigest() == fixture["member_sha256"]["dev"] and set(answers) == set(cases)
    data = {"fixture_sha256": digest(a.fixture), "answers": answers}
    a.output.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n")


def select(learned, baseline, truth):
    hybrid = baseline["hybrid"]
    if learned["fallback"]:
        assert learned["pool"] == hybrid
        picked = hybrid[:LIMIT]
    else:
        steps = learned["predicted_steps"]
        final_offset = 64 + sum(BUDGETS[steps][:-1])
        branch_choices = [*learned["read_from"], learned["pool"][final_offset]]
        picked = list(dict.fromkeys(branch_choices + learned["pool"]))[:LIMIT]
    assert len(picked) == len(set(picked)) == LIMIT and set(picked) <= set(learned["pool"])
    oracle = list(dict.fromkeys([i for i in learned["pool"] if i in truth] + learned["pool"]))[:LIMIT]
    return picked, oracle


def run(a, paragraphs, fixture):
    from functools import lru_cache

    import torch

    from run_kam_hotpot_pool_scaling import Reader
    from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT
    from run_kam_natural_feasibility import Encoder, measure_answer

    start = perf_counter()
    fx, cx = digest(a.fixture), fixture["corpus_sha256"]
    def packed(p):
        return json.loads(gzip.decompress(p.read_bytes()))
    baseline = packed(a.baseline)
    preliminary = list(map(packed, a.learned))
    revisions = json.loads((a.assets / "revisions.json").read_text())
    assert baseline["models"] == revisions and preliminary
    assert len(preliminary) == 3 or (a.joint_assignment and len(preliminary) == 6)
    assert all(p["models"] == revisions | {"google/flan-t5-base": BASE_REVISION} for p in preliminary)
    assert all(p["baseline_sha256"] == digest(a.baseline) for p in preliminary)
    assert all(p["fixture_sha256"] == fx and p["corpus_sha256"] == cx and (p["shard"], p["shards"]) == (a.shard, a.shards)
               for p in (baseline, *preliminary))
    answers = json.loads(a.labels.read_text())
    assert answers["fixture_sha256"] == fx
    fitted = None
    if a.joint_assignment:
        from analyze_kam_musique_assignment import evidence as assignment

    if a.selector_picks:
        sealed = json.loads(a.selector_picks.read_text())
        assert sealed["fixture_sha256"] == fx and sealed["policy_sha256"] == digest(a.selector_policy)
        assert sealed["provenance"]["corpus_sha256"] == cx
        assert sealed["provenance"]["weights_sha256"] == [p["weights_sha256"] for p in preliminary]
        assert [q["id"] for q in sealed["rows"]] == [q["id"] for q in fixture["rows"]]
        fitted = {q["id"]: q["selections"] for q in sealed["rows"]}
    model_dir = a.assets / f"reader_base_{BASE_REVISION}"
    assert digest(model_dir / "model.safetensors") == BASE_WEIGHT
    reader = Reader(model_dir)
    encoder = Encoder(a.assets / "encoder", 128)
    vectors = sorted((torch.load(p, weights_only=True, map_location="cpu") for p in a.vectors), key=lambda x: x["begin"])
    assert vectors[0]["begin"] == 0 and vectors[-1]["end"] == len(paragraphs)
    assert all(l["end"] == r["begin"] for l, r in zip(vectors, vectors[1:]))
    assert all(p["corpus_sha256"] == cx and p["models"] == revisions for p in vectors)
    bank = torch.cat([p["vectors"] for p in vectors], 0).cuda()
    cases = fixture["rows"][a.shard::a.shards]
    query = encoder([q["question"] for q in cases], "query")

    @lru_cache(maxsize=8192)
    def passage(i):
        p = paragraphs[i]
        tokens = reader.tokenizer(p["title"] + ". " + p["body"], add_special_tokens=False,
                                  max_length=PASSAGE_CAP, truncation=True)
        return reader.tokenizer.decode(tokens.input_ids)

    def exact_dense(vector, pool):
        values = bank[torch.tensor(pool, device="cuda")] @ vector
        order = values.argsort(descending=True, stable=True)[:LIMIT].tolist()
        return [pool[i] for i in order]

    rows, todo = [], []
    for j, (case, base, *learned) in enumerate(zip(cases, baseline["rows"], *(p["rows"] for p in preliminary), strict=True)):
        id_ = case["id"]
        assert all(r["id"] == id_ for r in (base, *learned))
        target = set(case["gold"])
        hybrid = base["hybrid"]
        raw = {"hybrid_rank": hybrid[:LIMIT], "hybrid_dense": exact_dense(query[j], hybrid),
               "hybrid_oracle": list(dict.fromkeys([i for i in hybrid if i in target] + hybrid))[:LIMIT],
               "gold_only": sorted(target), "no_evidence": []}
        for i, item in enumerate(learned):
            assert len(item["pool"]) == len(set(item["pool"])) == 128
            chosen, ceiling = select(item, base, target)
            raw[f"learned_path/{i}"] = chosen
            raw[f"learned_dense/{i}"] = exact_dense(query[j], item["pool"])
            raw[f"learned_oracle/{i}"] = ceiling
            if a.joint_assignment:
                paired, _ = assignment(item, chosen)
                raw[f"learned_assignment/{i}"] = paired["joint_assignment"]
                raw[f"learned_greedy/{i}"] = paired["greedy_subquery"]
            if fitted is not None:
                chosen = fitted[id_][i]
                assert len(chosen) == len(set(chosen)) == LIMIT and set(chosen) <= set(item["pool"])
                raw[f"learned_selector/{i}"] = chosen
        selected = {}
        for name, ids in raw.items():
            evidence = " ".join(passage(i) for i in ids)
            prompt = f"Answer the question with a short answer. Question: {case['question']} Evidence: {evidence}"
            record = {"evidence_ids": ids}
            selected[name] = record
            todo.append((prompt, answers["answers"][id_], record))
        rows.append({"id": id_, "arms": selected})
    prompts = list(dict.fromkeys(p for p, _, _ in todo))
    decoded = dict(zip(prompts, reader.answers(prompts), strict=True))
    for p, valid, outcome in todo:
        scores = [measure_answer(decoded[p], y) for y in valid]
        outcome.update({k: max(r[k] for r in scores) for k in ("em", "f1")})
    return {"fixture_sha256": fx, "corpus_sha256": cx, "baseline_sha256": digest(a.baseline),
            "learned_sha256": [digest(p) for p in a.learned], "label_sha256": digest(a.labels),
            "reader_revision": BASE_REVISION, "device": torch.cuda.get_device_name(),
            "shard": a.shard, "shards": a.shards, "unique_prompts": len(prompts),
            "elapsed_seconds": perf_counter() - start, "rows": rows} | (
                {"selector_picks_sha256": digest(a.selector_picks), "selector_policy_sha256": digest(a.selector_policy)}
                if fitted is not None else {})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "run"))
    for name in ("fixture", "output", "corpus"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in ("source", "assets", "baseline", "labels"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--learned", nargs="+", type=Path)
    p.add_argument("--selector-picks", type=Path)
    p.add_argument("--selector-policy", type=Path)
    p.add_argument("--joint-assignment", action="store_true")
    p.add_argument("--vectors", nargs=4, type=Path)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert (a.selector_picks is None) == (a.selector_policy is None)
    assert not (a.joint_assignment and a.selector_picks is not None)
    articles, fixture = corpus_and_fixture(a)
    if a.mode == "freeze":
        freeze(a, fixture)
        return
    import torch

    torch.set_float32_matmul_precision("highest")
    with torch.inference_mode():
        payload = run(a, articles, fixture)
    a.output.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
