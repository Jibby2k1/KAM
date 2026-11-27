#!/usr/bin/env python3
"""Separate reader size, passage capacity and answer-role selection on fixed MuSiQue evidence."""

import argparse
from functools import lru_cache
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest, words
from kam_flan_reader_assets import LARGE_REVISION, LARGE_WEIGHT
from run_kam_hotpot_pool_scaling import Reader
from run_kam_longmemeval_m_reader import window_start
from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT
from run_kam_musique_reader_composition import read_packed, unpack
from run_kam_native_musique import corpus_and_fixture
from run_kam_natural_feasibility import measure_answer

PLANS = {
    "archived_96": (96, "original", "all"),
    "prefix_96": (96, "head", "all"),
    "centered_96": (96, "center", "all"),
    "prefix_192": (192, "head", "all"),
    "centered_192": (192, "center", "all"),
    "first_only_192": (192, "head", "first"),
    "last_only_192": (192, "head", "last"),
}
READERS = {"base": (BASE_REVISION, BASE_WEIGHT), "large": (LARGE_REVISION, LARGE_WEIGHT)}


@torch.inference_mode()
def run(a, paragraphs, fixture):
    fx, cx = digest(a.fixture), fixture["corpus_sha256"]
    labels, spans, gold, primary = map(unpack, (a.labels, a.intermediate, a.gold_reference, a.primary_reference))
    assert all(x["fixture_sha256"] == fx for x in (labels, spans, gold, primary))
    assert gold["corpus_sha256"] == primary["corpus_sha256"] == cx
    assert primary["branch_labels_sha256"] == gold["label_sha256"] == digest(a.labels)
    assert primary["scoring_labels_sha256"] == digest(a.intermediate)
    models = {}
    for name, (revision, expected) in READERS.items():
        folder = a.assets / f"reader_{name}_{revision}"
        assert digest(folder / "model.safetensors") == expected
        models[name] = Reader(folder)
    tok = models["base"].tokenizer
    assert tok.get_vocab() == models["large"].tokenizer.get_vocab()
    assert tok.pad_token_id == models["large"].tokenizer.pad_token_id
    started = perf_counter()

    @lru_cache(maxsize=8192)
    def text(i, full):
        p = paragraphs[i]
        if full:
            return tok(p["title"]+". "+p["body"], add_special_tokens=False, truncation=True, max_length=96).input_ids
        return (tok(p["title"]+".", add_special_tokens=False, truncation=True, max_length=32).input_ids,
                tok(p["body"], add_special_tokens=False, verbose=False).input_ids)

    def rendered(i, width, mode, terms):
        if mode == "original":
            return tok.decode(text(i, True))
        title, full = text(i, False)
        budget = max(0, width - len(title))
        start = window_start(full, budget, terms) if mode == "center" else 0
        return tok.decode(title + full[start:start+budget])

    prompt_arms = {k: [] for k in PLANS}
    rows = []
    for q, g, r in zip(fixture["rows"][a.shard::a.shards], gold["rows"], primary["rows"], strict=True):
        id_ = q["id"]
        assert id_ == g["id"] == r["id"]
        if q["hops"] != 4:
            continue
        report = {"id": id_, "panels": {}}
        gold_control = g["arms"]["gold_only"]
        observed = r["arms"]["top1"]
        authors = spans["rows"][id_]["targets"]
        source = (("gold_paragraphs", gold_control["evidence_ids"], authors, bool(gold_control["em"])),
                  ("primary_selected", observed["fixed_evidence_ids"], observed["fixed_evidence_ids"], bool(observed["fixed_answer_em"])))
        terms = {i for w in set(words(q["question"])) if len(w) >= 4 for i in tok(w, add_special_tokens=False).input_ids}
        for case, context, directed, archived in source:
            assert len(context) == len(set(context)) == len(directed) == 4
            details = {}
            for plan, (cap, mode, role) in PLANS.items():
                selected = context if role == "all" else [directed[0 if role == "first" else -1]]
                evidence = " ".join(rendered(i, cap, mode, terms) for i in selected)
                prompt = f"Answer the question with a short answer. Question: {q['question']} Evidence: {evidence}"
                case_width = 512 if plan == "archived_96" else 1024
                ids = tok(prompt, max_length=case_width, truncation=True).input_ids
                prompt_arms[plan].append((tuple(ids), (1,)*len(ids)))
                details[plan] = {"evidence": selected, "tokens": len(ids),
                                 "prompt_sha256": sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
            report["panels"][case] = {"details": details, "archived_exact": archived, "judged": {name: {} for name in READERS}}
        rows.append(report)
    times = {m: {} for m in READERS}
    checked = 0
    for idx, (mode, prompts) in enumerate(prompt_arms.items()):
        order = tuple(models) if (idx+a.shard)%2 == 0 else tuple(reversed(models))
        for size in order:
            torch.cuda.synchronize()
            start = perf_counter()
            answers = read_packed(models[size], prompts)
            torch.cuda.synchronize()
            times[size][mode] = perf_counter() - start
            assert len(answers) == 2 * len(rows)
            for row, pair in zip(rows, zip(answers[::2], answers[1::2], strict=True), strict=True):
                for (group, item), answer in zip(row["panels"].items(), pair, strict=True):
                    got = bool(max(measure_answer(answer, value)["em"] for value in labels["answers"][row["id"]]))
                    if size == "base" and mode == "archived_96":
                        assert got == item["archived_exact"], (group, row["id"])
                        checked += 1
                    item["judged"][size][mode] = got
    return {"shard": a.shard, "shards": a.shards, "fixture_sha256": fx, "corpus_sha256": cx,
            "priors": {"gold": digest(a.gold_reference), "primary": digest(a.primary_reference)},
            "models": READERS, "elapsed_seconds": perf_counter()-started, "device": torch.cuda.get_device_name(),
            "times": times, "exact_archive_replays": checked, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("corpus", "fixture", "assets", "labels", "intermediate", "gold-reference", "primary-reference", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("shard", "shards"):
        p.add_argument("--" + k, type=int, required=True)
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    paragraphs, fixture = corpus_and_fixture(a)
    out = run(a, paragraphs, fixture)
    a.output.write_bytes(gzip.compress(json.dumps(out, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
