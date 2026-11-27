#!/usr/bin/env python3
"""Run paired scope changes with identical words over pinned Eurostat questions."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from analyze_kam_bright_transfer import limits, resample
from analyze_kam_eurostat_text_routing import archive_bank, asof
from audit_kam_natural_data import digest, words
from prepare_kam_eurostat_text_routing import COUNTRIES, NAMES, card, parse_dates, registry_country, registry_indicator
from run_kam_eurostat_text_routing import read, write
from run_kam_natural_feasibility import Encoder, batches

REQUEST = "Requested lookup: "
BACKGROUND = "Background lookup: "


def split_scope(message):
    parts = [(line.startswith(REQUEST), line[len(REQUEST):] if line.startswith(REQUEST) else line[len(BACKGROUND):])
             for line in message.splitlines() if line.startswith((REQUEST, BACKGROUND))]
    marked = [text for relevant, text in parts if relevant]
    return (marked[0] if len(marked) == 1 else None), [text for _, text in parts]


def matched_queries(fixture, split):
    q = fixture["questions"][split]
    matches = [r for r in q if r["kind"] == "answerable" and r["style"] in ("canonical", "paraphrase")]
    corpus = fixture["corpus"]
    change = defaultdict(dict)
    for r in matches:
        source = corpus[r["gold"]]
        address = source["country"], source["observation"]
        metric = source["indicator"]
        if r["style"] == "paraphrase":
            change["address", *address, r["cutoff"]][metric] = r
        else:
            change["cutoff", *address, metric][r["cutoff"]] = r
    paired = {}
    for key, p in change.items():
        assert len(p) == 2
        a, b = (p[k] for k in sorted(p))
        if a["gold"] == b["gold"]:
            continue
        paired[sha256(f"{a['id']}/{b['id']}".encode()).hexdigest()] = (key[0], (a, b))
    return paired


def pair_groups(fixture):
    paired = matched_queries(fixture, "test")
    # Each pair populates four marked rows. The truth changes independently of
    # the content order while the multiset of words remains fixed.
    rows = []
    for pair, (kind, (a, b)) in paired.items():
        signature = None
        for ordered in ((a, b), (b, a)):
            for target in (0, 1):
                message = "\n".join((REQUEST if i == target else BACKGROUND) + q["text"]
                                    for i, q in enumerate(ordered))
                bag = Counter(words(message))
                if signature is None:
                    signature = bag
                assert signature == bag
                gold = ordered[target]
                main, segments = split_scope(message)
                assert main == gold["text"] and len(segments) == 2
                rows.append({"pair": pair, "kind": kind, "position": target, "marked": True,
                             "message": message, "gold": gold["gold"]})
        message = "\n".join(BACKGROUND + r["text"] for r in (a, b))
        assert split_scope(message)[0] is None
        rows.append({"pair": pair, "kind": kind, "position": None, "marked": False,
                     "message": message, "gold": None})
    return rows


def score_dense(a, records, input_sha, revisions):
    candidates = sorted((torch.load(path, map_location="cpu", weights_only=True) for path in a.embeddings),
                        key=lambda row: row["span"][0])
    assert all(r["fixture_sha256"] == input_sha and r["model_revisions"] == revisions for r in candidates)
    assert candidates[0]["span"][0] == 0 and candidates[-1]["span"][1] == len(records)
    assert all(x["span"][1] == y["span"][0] for x, y in zip(candidates, candidates[1:]))
    bank = torch.cat([r["keys"] for r in candidates]).cuda()
    encoder = Encoder(a.assets/"encoder", 128)
    label = [(metric, geo) for metric in NAMES for geo in COUNTRIES]
    cards = encoder([card(metric, geo) for metric, geo in label], "passage")
    return encoder, bank, cards, label


@torch.inference_mode()
def routes(a, rows, docs, input_sha, revisions):
    encoder, bank, cards, label = score_dense(a, docs, input_sha, revisions)
    contents = {}
    for row in rows:
        active, segments = split_scope(row["message"])
        for text in (row["message"], *segments):
            contents.setdefault(text, len(contents))
        if active is not None:
            assert active in contents
    torch.cuda.synchronize()
    t = perf_counter()
    queries = encoder(list(contents), "query")
    card_rank = (queries @ cards.T).argmax(dim=1).tolist()
    dense_top = []
    for q in queries.split(128):
        dense_top.extend((q @ bank.T).argmax(dim=1).tolist())
    torch.cuda.synchronize()
    elapsed = perf_counter() - t
    history = archive_bank(docs)

    def resolve(text, clock_text, model):
        if clock_text is None:
            return None
        obs, cut = parse_dates(clock_text)
        if model == "literal":
            indicator, country = registry_indicator(text), registry_country(text)
        elif model == "card":
            indicator, country = label[card_rank[contents[text]]]
        else:
            record = docs[dense_top[contents[text]]]
            indicator, country = record["indicator"], record["country"]
        return asof(history, indicator, country, obs, cut)

    outputs = []
    for row in rows:
        active, (first, last) = split_scope(row["message"])
        chosen = {}
        for model in ("document", "card"):
            chosen[model] = {
                "whole_text_scope_dates": resolve(row["message"], active, model),
                "scoped_text_and_dates": resolve(active, active, model),
                "first_record": resolve(first, first, model),
                "last_record": resolve(last, last, model),
                "scoped_registry": resolve(active, active, "literal"),
                "whole_text_first_dates": resolve(row["message"], first, model),
                "whole_text_last_dates": resolve(row["message"], last, model),
            }
        outputs.append({k: v for k, v in row.items() if k != "message"} | {"routes": chosen})
    return elapsed, len(contents), outputs


def summarize(rows):
    summary = {}
    for kind in ("address", "cutoff"):
        outputs = {}
        tested = [r for r in rows if r["kind"] == kind and r["marked"]]
        blank = [r for r in rows if r["kind"] == kind and not r["marked"]]
        grouped = defaultdict(list)
        for i, row in enumerate(tested):
            grouped[row["pair"]].append(i)
        slices = list(grouped.values())
        assert all(len(x) == 4 for x in slices) and len(slices) == len(blank)
        sample = resample(slices)
        for mode in ("document", "card"):
            quality = {}
            old = np.asarray([r["routes"][mode]["whole_text_scope_dates"] == r["gold"] for r in tested], dtype=int)
            for policy in tested[0]["routes"][mode]:
                decisions = np.asarray([r["routes"][mode][policy] == r["gold"] for r in tested], dtype=int)
                quality[policy] = {"exact": int(sum(decisions)), "scope_pairs_all_four": sum(all(decisions[idx] for idx in group) for group in slices),
                                   "first_position_exact": sum(int(hit) for hit, r in zip(decisions, tested, strict=True) if r["position"] == 0),
                                   "last_position_exact": sum(int(hit) for hit, r in zip(decisions, tested, strict=True) if r["position"] == 1),
                                   "no_marker_refused": sum(r["routes"][mode][policy] is None for r in blank),
                                   "paired_vs_whole_text_points": float(100 * (decisions-old).mean()),
                                   "paired_vs_whole_text_interval_pp": [100 * v for v in limits(decisions-old, slices, sample)]}
            outputs[mode] = quality
        summary[kind] = {"marked_cases": len(tested), "paired_sets": len(slices), "no_marker_cases": len(blank), "policies": outputs}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("fixture", "assets", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--embeddings", type=Path, nargs=4, required=True)
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    fixture = read(a.fixture)
    revisions = read(a.assets/"revisions.json")
    signature = digest(a.fixture)
    pairs = pair_groups(fixture)
    seconds, query_variants, outcome = routes(a, pairs, fixture["corpus"], signature, revisions)
    write(a.output, {"fixture_sha256": signature, "device": torch.cuda.get_device_name(), "revisions": revisions,
                     "query_and_scan_seconds": seconds, "query_variants": query_variants,
                     "summary": summarize(outcome), "rows": outcome})


if __name__ == "__main__":
    main()
