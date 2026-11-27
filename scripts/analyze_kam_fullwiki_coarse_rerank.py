#!/usr/bin/env python3
"""Verify paired HotpotQA support and joint exact match after coarse search."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

SEARCH = ("full_rescore", "probe_64", "probe_256")
MODES = ("broad", "exact", *SEARCH)


def summarize(cases, initial, methods):
    counts = {name: Counter() for name in MODES}
    metrics = ("coverage", "shortlist", "support", "joint", "answer_em")
    flags = {name: {key: [] for key in metrics} for name in MODES}
    for q in cases:
        id_ = q["id"]
        gold = set(q["gold_ids"])
        original = initial[id_]
        for name, records in methods.items():
            row = records[id_]
            if name == "broad":
                top = original
                state = row["arms"]["broad"]
            else:
                hits = row["corpus_hits"]["conditioned_union"]
                top = original + [i for i in hits if i not in original]
                state = row["arms"]["conditioned_union"]
            pair = set(state["pair"])
            assert len(pair) == 2 and pair <= set(top)
            cover, shortlist, support, em = (gold <= set(top), state["gold_pair_in_shortlist"], pair == gold, state["exact"])
            assert support <= shortlist <= cover
            now = dict(zip(metrics, (cover, shortlist, support, support and em, em), strict=True))
            counts[name].update(now)
            for k, value in now.items():
                flags[name][k].append(value)
    return {"questions": len(cases), "counts": counts,
            "paired_vs_exact": {arm: {key: paired(flags[arm][key], flags["exact"][key]) for key in metrics}
                                 for arm in SEARCH}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arm in SEARCH:
        p.add_argument("--" + arm, nargs=4, required=True, type=Path)
    for name in ("reference", "fixture", "pools", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--hits", nargs=3, required=True, type=Path)
    a = p.parse_args()
    refs = sorted(unpack(a.reference), key=lambda c: c["shard"])
    fixture_bytes = gzip.decompress(a.fixture.read_bytes())
    fixture_sha = sha256(fixture_bytes).hexdigest()
    cases = json.loads(fixture_bytes)["rows"]["test"]
    initial = unpack(a.pools)
    anchor = initial["rows"]
    assert initial["fixture_sha256"] == fixture_sha
    locked = {q["id"]: q for c in refs for q in c["rows"]}
    collection = {"exact": locked, "broad": locked}
    metadata = ("fixture_sha256", "pools_sha256", "config_sha256", "index_sha256", "fitted_sha256", "heads_sha256",
                "ablations_sha256", "baseline_sha256", "models")
    parity = {}
    for arm, hit_file in zip(SEARCH, a.hits, strict=True):
        these = sorted(map(unpack, getattr(a, arm)), key=lambda c: c["shard"])
        assert [s["shard"] for s in these] == list(range(4))
        proposed = unpack(hit_file)
        assert proposed["search_method"] == arm
        hits_sha = sha256(hit_file.read_bytes()).hexdigest()
        verified = {}
        matches = Counter()
        for shard, prior in zip(these, refs, strict=True):
            assert all(shard[k] == prior[k] for k in metadata) and shard["corpus_hits_sha256"] == hits_sha
            assert shard["fixture_sha256"] == fixture_sha and shard["gpu"] == "NVIDIA L4" and shard["shards"] == 4
            for row, old in zip(shard["rows"], prior["rows"], strict=True):
                id_ = row["id"]
                assert id_ == old["id"] and row["first"] == old["first"] and row["pool_hash"] == old["pool_hash"]
                assert row["corpus_hits"]["conditioned_union"] == proposed["rows"][id_]["conditioned_union"]
                assert row["corpus_hits"]["unconditioned_union"] == old["corpus_hits"]["unconditioned_union"]
                matches.update({mode: row["arms"][mode] == old["arms"][mode]
                                for mode in ("broad", "unconditioned_union")})
                matches["conditioned_pair_exact"] += row["arms"]["conditioned_union"]["pair"] == old["arms"]["conditioned_union"]["pair"]
                matches["conditioned_outcome_exact"] += row["arms"]["conditioned_union"] == old["arms"]["conditioned_union"]
                verified[id_] = row
        collection[arm] = verified
        parity[arm] = matches
    total = {"all": cases}
    for is_complete in (True, False):
        name = "complete" if is_complete else "incomplete"
        total[name] = [q for q in cases if (set(q["gold_ids"]) <= set(anchor[q["id"]])) == is_complete]
    result = {"fixture_sha256": fixture_sha, "archive_pair_and_reader_parity": parity,
              "scopes": {name: summarize(group, anchor, collection) for name, group in total.items()}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
