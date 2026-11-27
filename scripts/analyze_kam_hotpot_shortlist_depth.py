#!/usr/bin/env python3
"""Audit paired shortlist depth, support recovery and decoding on fixed corpora."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

from analyze_kam_sentence_verifier import paired
from analyze_kam_shared_serving import unpack

DEPTHS = ("16", "32", "64")
MEASURES = ("shortlist", "support", "joint_em")
WORK = ("candidate_pairs", "candidate_documents", "distinct_sentences")
META = ("fixture_sha256", "pools_sha256", "config_sha256", "index_sha256", "fitted_sha256", "heads_sha256",
        "ablations_sha256", "baseline_sha256", "models", "indexed_documents")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    fields = {depth: {metric: [r[depth][metric] for r in rows] for metric in MEASURES} for depth in DEPTHS}
    return {"questions": len(rows),
            "totals": {p: {k: sum(v) for k, v in metrics.items()} for p, metrics in fields.items()},
            "paired": {d + "_vs_16": {m: paired(fields[d][m], fields["16"][m]) for m in MEASURES}
                       for d in DEPTHS[1:]} if rows else {},
            "work": {d: {k: {"mean": round(mean(r[d][k] for r in rows), 6),
                             "median": median(r[d][k] for r in rows)}
                         for k in WORK} for d in DEPTHS} if rows else {}}


def evaluate(profiles, study):
    fixture = gzip.decompress((study / "fixture.json.gz").read_bytes())
    cases = json.loads(fixture)["rows"]["test"]
    ids = [r["id"] for r in cases]
    assert len(ids) == len(set(ids)) == 1024
    by_id = {r["id"]: r for r in cases}
    pools = unpack(study / "pools-hybrid.json.gz")["rows"]
    reference = study / "hybrid-broad-replay.json.gz"
    archives = unpack(reference)
    ref = {r["id"]: r for s in archives for r in s["rows"]}
    assert set(ids) == ref.keys() == pools.keys()
    meta = {k: archives[0][k] for k in META}
    assert meta["fixture_sha256"] == sha256(fixture).hexdigest()
    assert meta["pools_sha256"] == digest(study / "pools-hybrid.json.gz")
    assert all(all(s[k] == v for k, v in meta.items()) for s in [*profiles, *archives])
    assert [s["shard"] for s in profiles] == list(range(4))
    assert all(s["reference_sha256"] == digest(reference) and s["shards"] == 4 for s in profiles)
    assert all(tuple(map(str, s["pair_depths_per_head"])) == DEPTHS for s in profiles)

    groups = {k: [] for k in ("all", "complete", "incomplete", "covered", "covered_not_shortlisted_at_16")}
    baseline = Counter()
    for shard in profiles:
        assert [r["id"] for r in shard["rows"]] == ids[shard["shard"]::4]
        for r in shard["rows"]:
            i = r["id"]
            old = ref[i]
            base = pools[i]
            gold = set(by_id[i]["gold_ids"])
            union = set(base) | set(old["corpus_hits"]["conditioned_union"])
            complete, covered = gold <= set(base), gold <= union
            assert r["expanded_size"] == len(union)
            assert r["pool_hash"] == old["pool_hash"] == sha256(
                json.dumps(base, separators=(",", ":")).encode()).hexdigest()
            assert set(r["arms"]) == set(DEPTHS)
            record = {}
            for d in DEPTHS:
                a = r["arms"][d]
                recovered = set(a["pair"]) == gold
                assert len(a["pair"]) == len(set(a["pair"])) == 2 and set(a["pair"]) <= union
                assert recovered <= a["gold_pair_in_shortlist"] <= covered
                assert int(d) <= a["candidate_pairs"] <= 2 * int(d)
                assert 2 <= a["candidate_documents"] <= len(union)
                assert a["distinct_sentences"] > 0 and type(a["exact"]) is bool
                record[d] = {"shortlist": a["gold_pair_in_shortlist"], "support": recovered,
                             "joint_em": recovered and a["exact"], **{k: a[k] for k in WORK}}
            for k in ("shortlist", *WORK):
                values = [record[d][k] for d in DEPTHS]
                assert values == sorted(values)
            original = r["arms"]["16"]
            for k in ("pair", "gold_pair_in_shortlist", "exact"):
                baseline[k] += original[k] == old["arms"]["conditioned_union"][k]
            baseline["complete_row"] += all(original[k] == old["arms"]["conditioned_union"][k]
                                             for k in ("pair", "gold_pair_in_shortlist", "exact"))
            groups["all"].append(record)
            groups["complete" if complete else "incomplete"].append(record)
            if covered:
                groups["covered"].append(record)
                if not record["16"]["shortlist"]:
                    groups["covered_not_shortlisted_at_16"].append(record)
    return {**meta, "b200_reference_sha256": digest(reference), "replay_gpus": [s["gpu"] for s in profiles],
            "original_16_replay_parity": baseline, "cohorts": {k: summarize(v) for k, v in groups.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", type=Path, nargs=4)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = evaluate([unpack(f) for f in a.profiles], a.study)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
