#!/usr/bin/env python3
"""Freeze a disjoint development panel and auditable token spans in locked prompts."""
import argparse
import json
from pathlib import Path
import re

from kam_adapter_preservation import digest
from kam_math_kl_protocol import write_json
from prepare_kam_long_sparse import CONTEXT, display, freeze_retrieval, load_locked_tokenizer, retrieval_segments, tokenize_retrieval
from prepare_kam_natural_router import title_key

DEV_CASES = 48
PROTOCOL = "kam-evidence-fallback-v1"


def read(path):
    return json.loads(path.read_text())


def freeze(source, prior_reservations, reference, work):
    frozen = read(reference / "retrieval.json")
    assert frozen["source_sha256"] == digest(source)
    old = frozen["cases"]
    all_ids = set(prior_reservations.read_text().split()) | {r["id"] for r in old}
    titles = {title_key(r[side]["title"]) for r in old for side in ("earlier", "gold")}
    families = {r["family"] for r in old}
    work.mkdir(parents=True, exist_ok=True)
    reserve = work / "dev-reservations.txt"
    reserve.write_text("\n".join(sorted(all_ids)) + "\n")
    freeze_retrieval(source, reserve, work, count=DEV_CASES, excluded_titles=titles, excluded_families=families)
    path = work / "retrieval.json"
    path.rename(work / "dev-retrieval.json")


def token_span(offsets, start, end, offset=0):
    indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and a != b]
    assert indices and indices == list(range(indices[0], indices[-1]+1))
    return [offset+indices[0], offset+indices[-1]+1]


def locate(tokenizer, raw, prompts):
    index = {r["id"]: r for r in raw["cases"]}
    evidence = []
    for prompt in prompts:
        row = index[prompt["id"]]
        earlier, gold, foil = [tokenizer.decode(tokenizer.encode(display(row[k]),
            add_special_tokens=False, truncation=True, max_length=256)) for k in ("earlier", "gold", "foil")]
        endpoints = (gold, foil) if prompt["expected"] == 0 else (foil, gold)
        left, right = retrieval_segments(tokenizer, row, earlier, endpoints)
        a, b = [tokenizer(t, add_special_tokens=False, return_offsets_mapping=True) for t in (left, right)]
        ids = prompt["ids"]
        assert len(ids) == CONTEXT and ids[:len(a.input_ids)] == a.input_ids and ids[-len(b.input_ids):] == b.input_ids
        prefix = "\n\nEarlier passage:\n"
        prior_position = left.index(prefix) + len(prefix)
        assert left[prior_position:prior_position+len(earlier)] == earlier
        position_a = len("\n\nPassage A:\n")
        position_b = position_a+len(endpoints[0])+len("\n\nPassage B:\n")
        for p, text in zip((position_a, position_b), endpoints, strict=True):
            assert right[p:p+len(text)] == text
        ps = token_span(a.offset_mapping, prior_position, prior_position+len(earlier))
        span_a, span_b = [token_span(b.offset_mapping, p, p+len(t), CONTEXT-len(b.input_ids))
                         for p,t in zip((position_a, position_b), endpoints, strict=True)]
        mentions = list(re.finditer(r"(?<!\w)" + re.escape(row["anchor"]) + r"(?!\w)", earlier, re.IGNORECASE))
        assert mentions
        anchor = token_span(a.offset_mapping, prior_position+mentions[0].start(), prior_position+mentions[0].end())
        assert ps[0] <= anchor[0] < anchor[1] <= ps[1] < min(span_a[0], span_b[0])
        gold_span, foil_span = (span_a, span_b) if prompt["expected"] == 0 else (span_b, span_a)
        evidence.append({"id": prompt["id"], "family": prompt["family"], "expected": prompt["expected"],
                         "earlier": ps, "gold": gold_span, "foil": foil_span, "bridge": anchor})
    return evidence


def encode(root, reference, work):
    tokenizer, tokenizer_path = load_locked_tokenizer(root)
    raw = read(work / "dev-retrieval.json")
    dev, dev_cases = tokenize_retrieval(tokenizer, raw, DEV_CASES)
    token_path = work / "dev-inputs.json"
    write_json(token_path, dev)
    base_path = reference / "inputs.json"
    manifest_path = reference / "manifest.json"
    prior = read(manifest_path)
    assert digest(base_path) == prior["inputs_sha256"]
    holds = read(base_path)["retrieval"]
    assert len(holds) == 640
    assert not ({r["id"] for r in dev} & {r["id"] for r in holds})
    assert not ({r["family"] for r in dev} & {r["family"] for r in holds})
    write_json(work / "spans.json", {"protocol": PROTOCOL, "holdout_manifest_sha256": digest(manifest_path),
        "tokenizer_sha256": digest(tokenizer_path), "dev_inputs_sha256": digest(token_path),
        "dev_raw_sha256": digest(work / "dev-retrieval.json"), "source_sha256": raw["source_sha256"],
        "reservations_sha256": raw["reserved_sha256"], "dev_cases": dev_cases,
        "dev": locate(tokenizer, raw, dev), "holdout": locate(tokenizer, read(reference / "retrieval.json"), holds)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    cli = p.add_subparsers(dest="command", required=True)
    build = cli.add_parser("freeze")
    for name in ("source", "prior_reservations", "reference", "work"):
        build.add_argument(name, type=Path)
    tokens = cli.add_parser("encode")
    for name in ("root", "reference", "work"):
        tokens.add_argument(name, type=Path)
    a = p.parse_args()
    if a.command == "freeze":
        freeze(a.source, a.prior_reservations, a.reference, a.work)
    else:
        encode(a.root, a.reference, a.work)
