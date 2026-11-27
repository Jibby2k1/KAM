#!/usr/bin/env python3
"""Measure answer dependence on the author-identified terminal passage."""

import argparse
from functools import lru_cache
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re

import torch

from audit_kam_natural_data import digest
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_reader_composition import read_packed, unpack
from run_kam_musique_reader_windows import READERS
from run_kam_native_musique import corpus_and_fixture
from run_kam_natural_feasibility import measure_answer, normalized_answer


def scored(answer, alternatives):
    return any(measure_answer(answer, value)["em"] for value in alternatives)


def occurrence(value):
    return re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", re.IGNORECASE)


def shape(answer):
    if re.search(r"\d", answer):
        return re.sub(r"\d", "#", answer.lower()) if re.fullmatch(r"[\d .,\-–/%]+", answer) else None
    parts = answer.split()
    if not parts or not all(re.fullmatch(r"[A-Za-z][A-Za-z'’\-]*", p) for p in parts):
        return None
    return tuple("upper" if p.isupper() else "title" if p[0].isupper() else "lower" for p in parts)


def run(a, passages, fixture):
    answers, labels, archive = map(unpack, (a.labels, a.intermediate, a.window_reference))
    stamp = digest(a.fixture)
    assert all(x["fixture_sha256"] == stamp for x in (answers, labels, archive))
    assert archive["corpus_sha256"] == fixture["corpus_sha256"]
    assert archive["shard"] == a.shard and archive["shards"] == a.shards
    readers = {}
    for size, (rev, weight) in READERS.items():
        path = a.assets / f"reader_{size}_{rev}"
        assert digest(path / "model.safetensors") == weight
        readers[size] = Reader(path)
    tok = readers["base"].tokenizer
    assert tok.get_vocab() == readers["large"].tokenizer.get_vocab()

    @lru_cache(maxsize=8192)
    def tokens(text):
        return tok(text, add_special_tokens=False, verbose=False).input_ids

    @lru_cache(maxsize=2048)
    def render(i):
        doc = passages[i]
        title = tokens(doc["title"] + ".")[:32]
        body = tokens(doc["body"])[:192-len(title)]
        return tok.decode(title), tok.decode(body), tok.decode(title + body)

    def prompt(q, evidence):
        string = f"Answer the question with a short answer. Question: {q} Evidence: {' '.join(evidence)}"
        ids = tuple(tok(string, truncation=True, max_length=1024).input_ids)
        return ids, (1,)*len(ids)

    pool = sorted({v for q in fixture["rows"] if q["hops"] == 4 for v in answers["answers"][q["id"]]
                   if len(v.strip()) >= 3 and shape(v.strip()) is not None})
    examples = (q for q in fixture["rows"][a.shard::a.shards] if q["hops"] == 4)
    pending, output, truth = {}, [], []
    for q, old in zip(examples, archive["rows"], strict=True):
        id_ = q["id"]
        assert old["id"] == id_
        control = old["panels"]["gold_paragraphs"]
        ids = control["details"]["prefix_192"]["evidence"]
        end = labels["rows"][id_]["targets"][-1]
        assert set(ids) == set(q["gold"]) and ids.count(end) == 1
        title, body, full = render(end)
        text = q["question"]
        earlier = [render(i)[-1] for i in ids if i != end]
        views = {"question_only": [], "earlier_three": earlier, "terminal_title": [title],
                 "terminal_body": [body], "terminal_full": [full], "all": [render(i)[-1] for i in ids]}
        aliases = answers["answers"][id_]
        hits = {key: any(occurrence(v).search(p) for v in aliases for p in surface)
                for key, surface in {"title": [title], "body": [body], "earlier": earlier, "question": [text]}.items()}
        donor = None
        direct = [v for v in aliases if len(v) >= 3 and occurrence(v).search(full)]
        if len({v.lower() for v in direct}) == 1 and not hits["earlier"] and not hits["question"]:
            target = direct[0]
            family = shape(target)
            combined = " ".join([text, *earlier, full])
            if family is not None:
                former = {tuple(normalized_answer(x)) for x in aliases}
                candidates = [v for v in pool if shape(v) == family and len(tokens(v)) == len(tokens(target))
                              and tuple(normalized_answer(v)) not in former
                              and not occurrence(v).search(combined)]
                if candidates:
                    donor = candidates[int(sha256(id_.encode()).hexdigest()[:8], 16) % len(candidates)]
                    changed = occurrence(target).sub(lambda _: donor, full)
                    assert occurrence(donor).search(changed) and not occurrence(target).search(changed)
                    views["terminal_replaced"] = [changed]
                    views["all_terminal_replaced"] = [changed if i == end else render(i)[-1] for i in ids]
        prompts = {k: prompt(text, v) for k, v in views.items()}
        for mode, source in (("terminal_full", "last_only_192"), ("all", "prefix_192")):
            check = sha256(json.dumps(list(prompts[mode][0]), separators=(",", ":")).encode()).hexdigest()
            assert check == control["details"][source]["prompt_sha256"], (id_, mode)
        for key, item in prompts.items():
            pending.setdefault(key, []).append((len(output), item))
        output.append({"id": id_, "surface_answer_occurrence": hits, "rewrite_eligible": donor is not None,
                       "judged": {s: {} for s in readers}})
        truth.append((aliases, donor, control["judged"]))
    for size, reader in readers.items():
        for mode, docs in pending.items():
            replies = read_packed(reader, [ids for _, ids in docs])
            for (i, _), reply in zip(docs, replies, strict=True):
                aliases, donor, old = truth[i]
                gold = scored(reply, aliases)
                if mode in ("terminal_full", "all"):
                    key = "last_only_192" if mode == "terminal_full" else "prefix_192"
                    assert gold == old[size][key], (size, key, output[i]["id"])
                output[i]["judged"][size][mode] = {"original_exact": gold,
                                                    "donor_exact": bool(donor and scored(reply, [donor]))}
    return {"fixture_sha256": stamp, "window_reference_sha256": digest(a.window_reference), "models": READERS,
            "device": torch.cuda.get_device_name(), "shard": a.shard, "shards": a.shards, "rows": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("corpus", "fixture", "assets", "labels", "intermediate", "window-reference", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    for field in ("shard", "shards"):
        p.add_argument("--" + field, type=int, required=True)
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    data, fixture = corpus_and_fixture(a)
    result = run(a, data, fixture)
    a.output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
