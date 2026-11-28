#!/usr/bin/env python3
"""Fit title/full-passage BM25 fusion using source-disjoint MuSiQue questions."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_kam_natural_data import digest, rank, stable, words
from kam_math_kl_protocol import write_json
from prepare_kam_musique_endpoint_pairs import candidate
from prepare_kam_native_musique import MEMBERS, canonical_paragraph
from prepare_kam_natural_router import title_key
from summarize_kam_pretrained_endpoint_selection import family_difference

PROTOCOL = "kam-musique-title-fusion-v1"
WEIGHTS = (0., .125, .25, .5, 1., 2., 4.)
RRF_OFFSET = 4


def scores(question, documents, gold, weight_grid=WEIGHTS):
    outcomes = {w: [] for w in weight_grid}
    for reverse in (False, True):
        ordered = documents[::-1] if reverse else documents
        n = len(ordered)
        expected = n - 1 - gold if reverse else gold
        passage = rank(question, [title + ". " + body for title, body in ordered], limit=n)
        titles = rank(question, [title for title, _ in ordered], limit=n)
        text_rr = {entry: 1 / (RRF_OFFSET + i + 1) for i, entry in enumerate(passage)}
        title_rr = {entry: 1 / (RRF_OFFSET + i + 1) for i, entry in enumerate(titles)}
        for w in weight_grid:
            combined = sorted(range(n), key=lambda i: (-(text_rr[i] + w * title_rr[i]), i))
            outcomes[w].append(combined.index(expected) + 1)
    return outcomes


def load_archives(source):
    data = {}
    with ZipFile(source) as archive:
        for split, member in MEMBERS.items():
            with archive.open(member) as contents:
                data[split] = [json.loads(line) for line in contents]
    return data


def corpus_rows(source, split, reserved_rows, exclude, weights):
    reserved_ids = {r["id"] for r in reserved_rows}
    reserved_words = {tuple(words(r["question"])) for r in reserved_rows}
    reserved_family = {stable(r["family"]) for r in reserved_rows}
    reserved_titles = {title_key(p["title"]) for r in reserved_rows for p in r["evidence"]}
    out, included = [], []
    for row in sorted(source[split], key=lambda r: stable(PROTOCOL + split + r["id"])):
        info = candidate(row)
        if not info or info["hops"] != 2 or info["id"] in reserved_ids:
            continue
        wording, group = tuple(words(info["question"])), stable(info["family"])
        titles = {title_key(p["title"]) for p in info["evidence"]}
        if (int(group[:8], 16) % 2 != int(split == "dev") or wording in reserved_words or
                group in reserved_family or titles & reserved_titles or group in exclude):
            continue
        added = {}
        supports = {p["id"] for p in info["evidence"]}
        for paragraph in row["paragraphs"]:
            ident = stable("\x1f".join(canonical_paragraph(paragraph)))
            if not paragraph["is_supporting"] and ident not in supports:
                added[ident] = paragraph["title"], paragraph["paragraph_text"]
        if len(added) < 16:
            continue
        first = info["evidence"][0]
        order = sorted(added, key=lambda ident: stable(PROTOCOL + row["id"] + ident))
        docs = [(first["title"], first["body"])] + [added[ident] for ident in order]
        out.append({"id": row["id"], "family": group, "ranks": scores(info["question"], docs, 0, weights)})
        included.append(info)
        reserved_words.add(wording)
        reserved_titles |= titles
        if len(out) == (800 if split == "train" else 500):
            break
    return out, included


def report(rows, selected):
    families = [row["family"] for row in rows]
    thresholds = (1, 2, 4)
    hits = {w: {k: np.array([[max(r["ranks"][w]) <= k for r in rows]]) for k in thresholds}
            for w in rows[0]["ranks"]}
    outcomes = {str(w): {f"top_{k}": int(flags.sum()) for k, flags in by_k.items()} for w, by_k in hits.items()}
    contrast = {k: family_difference(hits[selected][k], hits[0.][k], families) for k in thresholds}
    return {"n": len(rows), "families": len(set(families)), "recall_in_both_orders": outcomes,
            "selected_minus_plain": contrast, "records": rows}


def audit(source, raw_panel, existing_panel, approval):
    registration = json.loads(approval.read_text())
    assert registration["protocol"] == PROTOCOL and registration["source_sha256"] == digest(source)
    assert registration["earlier_panel_sha256"] == digest(existing_panel) and registration["live_panel_sha256"] == digest(raw_panel)
    assert tuple(registration["title_weights"]) == WEIGHTS and registration["rrf_offset"] == RRF_OFFSET
    archived = json.loads(raw_panel.read_text())
    earlier = json.loads(existing_panel.read_text())
    reserved = [q for pair in [*archived, *earlier] for q in pair["questions"]]
    dataset = load_archives(source)
    train, included = corpus_rows(dataset, "train", reserved, set(), WEIGHTS)
    selected = max(WEIGHTS, key=lambda w: sum(max(row["ranks"][w]) == 1 for row in train))
    held_weights = (0., selected) if selected else (0.,)
    heldout, _ = corpus_rows(dataset, "dev", reserved + included, {r["family"] for r in train}, held_weights)
    assert len(train) >= 100 and len(heldout) >= 50
    return {"manifest_sha256": digest(approval), "selected_title_weight": selected,
            "fit": report(train, selected), "heldout": report(heldout, selected)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "live_panel", "earlier_panel", "manifest", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write_json(args.output, audit(args.source, args.live_panel, args.earlier_panel, args.manifest))
