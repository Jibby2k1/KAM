#!/usr/bin/env python3
"""Audit paired native 2Wiki retrieval and fixed-article reader controls."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

SEARCH_METHODS = ("bm25", "dense", "hybrid", "conditioned")
READ_METHODS = ("native_leads", "supplied_context", "annotated_sentences")
GROUPS = ("exact", "partial")


def load_shards(paths, fixture):
    shards = sorted(map(unpack, paths), key=lambda r: r["shard"])
    assert [s["shard"] for s in shards] == list(range(4))
    assert all(s["shards"] == 4 and s["device"] == "NVIDIA L4" and s["corpus"] == fixture["corpus"]
               and s["fixture_sha256"] == fixture["fixture_sha256"] for s in shards)
    for s in shards:
        assert [r["id"] for r in s["rows"]] == [r["id"] for r in fixture["rows"][s["shard"]::4]]
    return {r["id"]: r for s in shards for r in s["rows"]}


def verify_fixture(fixture):
    rows = fixture["rows"]
    assert len(rows) == len({r["id"] for r in rows}) == 1024
    assert len({u["title_sha256"] for r in rows for u in r["units"]}) == 2048
    assert len({u["corpus_id"] for r in rows for u in r["units"]}) == 2048
    subsets = defaultdict(dict)
    for case in rows:
        units = case["units"]
        assert len({u["corpus_id"] for u in units}) == 2 and all(u["indexed"] for u in units)
        assert all(u["exact_unit"] for u in units) == (case["cohort"] == "exact")
        assert case["cohort"] not in subsets[case["match_id"]]
        subsets[case["match_id"]][case["cohort"]] = (case["type"], len(units))
    assert len(subsets) == 512 and all(set(v) == set(GROUPS) and v["exact"] == v["partial"] for v in subsets.values())
    assert Counter(x["exact"][0] for x in subsets.values()) == {"comparison": 192, "compositional": 192, "inference": 128}


def retrieval(cases, results):
    counts = defaultdict(Counter)
    vectors = defaultdict(list)
    for case in cases:
        data = results[case["id"]]
        units = case["units"]
        gold = {u["corpus_id"] for u in units}
        assert data["conditioned"][:64] == data["dense"][:64]
        for method in SEARCH_METHODS:
            ranks = data[method]
            assert len(ranks) == len(set(ranks)) == 128
            for cutoff in (16, 64, 128):
                contained = set(ranks[:cutoff])
                complete = gold <= contained
                strict = sum(u["corpus_id"] in contained and u["exact_unit"] for u in units)
                excerpts = sum(u["corpus_id"] in contained and u["shared_excerpt"] for u in units)
                key = f"{method}/{cutoff}"
                counts[key].update({"title_complete": complete, "gold_titles": len(gold & contained),
                                    "all_exact_units": strict == len(units), "exact_units": strict,
                                    "all_units_have_excerpt": excerpts == len(units), "units_with_excerpt": excerpts})
                vectors[key].append(complete)
    return {"questions": len(cases), "annotation_units": sum(len(x["units"]) for x in cases),
            "retrieval": dict(counts),
            "paired_title_complete": {
                f"{m}_vs_dense128": paired(vectors[f"{m}/128"], vectors["dense/128"])
                for m in ("bm25", "hybrid", "conditioned")
            }}


def readers(cases, results):
    totals = defaultdict(Counter)
    em, visible = defaultdict(list), defaultdict(list)
    for case in cases:
        completed = results[case["id"]]
        units = len(case["units"])
        assert units == completed["units"]
        assert set(completed["arms"]) == set(READ_METHODS)
        for method, outcome in completed["arms"].items():
            count = outcome["visible_units"]
            assert 0 <= count <= units and outcome["em"] in (0.0, 1.0)
            complete = count == units
            em[method].append(int(outcome["em"]))
            visible[method].append(complete)
            if method == "native_leads" and case["cohort"] == "partial":
                # Sentence text may be normalized by the reader tokenizer, so
                # corpus matching and prompt matching are audited independently.
                totals[method]["matches_initial_corpus_exact"] += complete == all(u["exact_unit"] for u in case["units"])
            totals[method].update({"em": int(outcome["em"]), "all_annotation_visible": complete,
                                   "visible_units": count, "visibility_and_em": int(outcome["em"]) * complete})
    return {"questions": len(cases), "reader": dict(totals),
            "paired_em": {f"{m}_vs_native": paired(em[m], em["native_leads"]) for m in READ_METHODS[1:]},
            "paired_all_annotation_visible": {f"{m}_vs_native": paired(visible[m], visible["native_leads"])
                                               for m in READ_METHODS[1:]}}


def prompt_audit(cases, reading, measurements):
    totals = defaultdict(Counter)
    for case in cases:
        i = case["id"]
        unit_count = len(case["units"])
        assert unit_count == measurements[i]["units"]
        for mode, outcome in measurements[i]["arms"].items():
            assert mode in READ_METHODS
            assert outcome["raw_reference_visible"] == reading[i]["arms"][mode]["visible_units"]
            assert all(0 <= value <= unit_count for value in outcome.values())
            for field, count in outcome.items():
                totals[mode][field + "_units"] += count
                totals[mode][field + "_complete"] += count == unit_count
    return dict(totals)


def summarize(fixture, search, reading, prompt=None):
    verify_fixture(fixture)
    cases = fixture["rows"]
    strata = {"all": cases, **{k: [r for r in cases if r["cohort"] == k] for k in GROUPS}}
    for label in GROUPS:
        incomplete = [r for r in strata[label] if not {u["corpus_id"] for u in r["units"]} <= set(search[r["id"]]["dense"][:64])]
        strata[f"{label}/dense64_incomplete"] = incomplete
        for has_gold in (True, False):
            anchor = "gold" if has_gold else "other"
            strata[f"{label}/dense64_incomplete/{anchor}_anchor"] = [
                r for r in incomplete if (search[r["id"]]["dense"][0] in {u["corpus_id"] for u in r["units"]}) == has_gold
            ]
    retrieval_results = {k: retrieval(rows, search) for k, rows in strata.items() if rows}
    reader_results = {}
    if reading is not None:
        groups = {k: strata[k] for k in ("all", *GROUPS)}
        groups |= {f"{label}/{kind}": [r for r in strata[label] if reading[r["id"]]["answer_kind"] == kind]
                   for label in GROUPS for kind in ("binary", "other")}
        reader_results = {k: readers(rows, reading) for k, rows in groups.items() if rows}
    if prompt is not None:
        assert reading is not None and prompt["fixture_sha256"] == fixture["fixture_sha256"] and prompt["corpus"] == fixture["corpus"]
        assert prompt["source_sha256"] == fixture["source_sha256"]
        assert [x["id"] for x in prompt["rows"]] == [x["id"] for x in cases]
        lookup = {x["id"]: x for x in prompt["rows"]}
        for label in GROUPS:
            reader_results[label]["tokenizer_audit"] = prompt_audit(strata[label], reading, lookup)
    return {"fixture_sha256": fixture["fixture_sha256"], "corpus": fixture["corpus"],
            "retrieval": retrieval_results, "fixed_article_reader": reader_results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search", nargs=4, required=True, type=Path)
    p.add_argument("--reading", nargs=4, type=Path)
    p.add_argument("--prompt-audit", type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path)
    source.add_argument("--manifest", type=Path)
    p.add_argument("--write-manifest", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.fixture:
        raw = a.fixture.read_bytes()
        fixture = json.loads(raw)
        fixture["fixture_sha256"] = sha256(raw).hexdigest()
        for row in fixture["rows"]:
            del row["question"]
        if a.write_manifest:
            a.write_manifest.write_bytes(gzip.compress(json.dumps(fixture, separators=(",", ":")).encode(), mtime=0))
    else:
        fixture = unpack(a.manifest)
    search = load_shards(a.search, fixture)
    reading = load_shards(a.reading, fixture) if a.reading else None
    result = summarize(fixture, search, reading, unpack(a.prompt_audit) if a.prompt_audit else None)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
