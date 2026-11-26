#!/usr/bin/env python3
"""Compare first-stage affinity and broad-ranked expansion anchors on frozen pools."""

import argparse
from hashlib import sha256
import gzip
import json
from pathlib import Path

from analyze_kam_sentence_verifier import paired

ARMS = ("broad", "affinity_anchor", "broad_anchor")
METRICS = ("support", "joint_em", "coverage", "shortlist")


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def load_shards(source, marker):
    return ([unpack(p) for p in sorted(source.glob(marker + "-*.json.gz"))]
            if source.is_dir() else unpack(source))


def count(rows):
    result = {"questions": len(rows)}
    metrics = {arm: {m: [r[arm][m] for r in rows] for m in METRICS} for arm in ARMS}
    result["arm_totals"] = {a: {m: sum(values) for m, values in metric.items()} for a, metric in metrics.items()}
    result["anchor_correct"] = {a: sum(r[a]["anchor_correct"] for r in rows) for a in ARMS[1:]}
    result["mean_unique_additions"] = {a: round(sum(r[a]["new"] for r in rows) / len(rows), 6) if rows else None
                                       for a in ARMS[1:]}
    result["anchor_changed"] = sum(r["anchor_changed"] for r in rows)
    if rows:
        result["paired"] = {a + "_vs_" + b: {m: paired(metrics[a][m], metrics[b][m]) for m in METRICS}
                            for a, b in (("broad_anchor", "affinity_anchor"), ("broad_anchor", "broad"),
                                         ("affinity_anchor", "broad"))}
    return result


def evaluate(a):
    old, new = unpack(a.affinity), load_shards(a.input, "broad")
    frozen = unpack(a.pools)
    pool_digest = sha256(a.pools.read_bytes()).hexdigest()
    base = gzip.decompress(a.fixture.read_bytes())
    assert frozen["fixture_sha256"] == sha256(base).hexdigest()
    anchors = sha256(a.affinity.read_bytes()).hexdigest()
    keys = ("fixture_sha256", "pools_sha256", "config_sha256", "index_sha256", "fitted_sha256", "heads_sha256",
            "ablations_sha256", "baseline_sha256", "models", "indexed_documents", "guard_columns", "split")
    provenance = {key: old[0][key] for key in keys}
    assert all(all(s[k] == v for k, v in provenance.items()) for s in (*old, *new))
    assert provenance["fixture_sha256"] == frozen["fixture_sha256"] and provenance["pools_sha256"] == pool_digest
    assert provenance["index_sha256"] == frozen["index_sha256"]
    for group in (old, new):
        assert len(group) == 4 and sorted(s["shard"] for s in group) == list(range(4))
    assert all(s["anchor_reference_sha256"] == anchors for s in new)
    candidates = {q["id"]: set(q["gold_ids"]) for q in json.loads(base)["rows"]["test"]}
    before, after = ({r["id"]: r for s in group for r in s["rows"]} for group in (old, new))
    assert before.keys() == after.keys() == candidates.keys() == frozen["rows"].keys()
    results = []
    matches = 0
    for id_ in sorted(candidates):
        pool = frozen["rows"][id_]
        assert len(pool) == len(set(pool)) == 64
        original, gold = set(pool), candidates[id_]
        assert len(gold) == 2
        left, right = before[id_], after[id_]
        assert left["pool_hash"] == right["pool_hash"] == sha256(
            json.dumps(pool, separators=(",", ":")).encode()).hexdigest()
        assert right["first"] == left["arms"]["broad"]["pair"][0] and left["first"] in original
        matches += left["arms"]["broad"] == right["arms"]["broad"]
        row = {"missing": len(gold - original), "anchor_changed": left["first"] != right["first"]}
        for arm, source in (("broad", left), ("affinity_anchor", left), ("broad_anchor", right)):
            added = source["corpus_hits"]["conditioned_union"] if arm != "broad" else []
            if added:
                assert len(added) == len(set(added)) == 64 and source["first"] not in added
            scored = source["arms"]["broad" if arm == "broad" else "conditioned_union"]
            pair = set(scored["pair"])
            union = original | set(added)
            support = pair == gold
            assert len(pair) == 2 and pair <= union
            assert support <= scored["gold_pair_in_shortlist"] <= (gold <= union)
            row[arm] = {"coverage": gold <= union, "shortlist": scored["gold_pair_in_shortlist"],
                        "support": support, "joint_em": support and scored["exact"],
                        "anchor_correct": source["first"] in gold, "new": len(union - original)}
        results.append(row)
    assert len(results) == 1024
    groups = {"all": results, "complete_original": [r for r in results if r["missing"] == 0],
              "incomplete_original": [r for r in results if r["missing"] > 0],
              "one_missing": [r for r in results if r["missing"] == 1],
              "two_missing": [r for r in results if r["missing"] == 2],
              "different_anchor": [r for r in results if r["anchor_changed"]]}
    summary = {**provenance, "affinity_replay_sha256": anchors, "repeat_broad_matches": matches,
               "gpu_shards": {"affinity": [s["gpu"] for s in old], "broad_anchor": [s["gpu"] for s in new]},
               "cohorts": {k: count(v) for k, v in groups.items()}}
    return summary, new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    for k in ("affinity", "fixture", "pools", "output"):
        parser.add_argument("--" + k, type=Path, required=True)
    parser.add_argument("--write-replay", type=Path)
    a = parser.parse_args()
    result, new = evaluate(a)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        a.write_replay.write_bytes(gzip.compress(json.dumps(new, separators=(",", ":"), allow_nan=False).encode(),
                                                 mtime=0))


if __name__ == "__main__":
    main()
