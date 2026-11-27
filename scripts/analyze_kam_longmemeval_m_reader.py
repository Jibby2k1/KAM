#!/usr/bin/env python3
"""Audit frozen LongMemEval answer outcomes and the effect of showing dates."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from run_kam_longmemeval_m_reader import ARMS

CONTRASTS = (("dense_dated", "dense"), ("user", "dense"), ("marked_in_dense", "dense"),
             ("oracle_marked", "marked_in_dense"), ("oracle_marked_dated", "oracle_marked"))


def summary(cases, saved):
    groups = shared_evidence_clusters([{"gold": q["gold_ids"] or [q["id"]]} for q in cases])
    stats, binary = {}, {}
    for arm in ARMS:
        rows = [saved[q["id"]][arm] for q in cases]
        binary[arm] = {key: [bool(r[key]) for r in rows] for key in ("em", "abstained", "marked_support")}
        binary[arm]["joint"] = [r["em"] and r["marked_support"] for r in rows]
        stats[arm] = {key: sum(hits) for key, hits in binary[arm].items()}
        stats[arm].update({"mean_f1": mean(r["f1"] for r in rows),
                           "median_prompt_tokens": median(r["prompt_tokens"] for r in rows),
                           "median_retained_turn_tokens": median(r["retained_turn_tokens"] for r in rows),
                           "questions_with_clipped_turns": sum(r["clipped_turns"] > 0 for r in rows)})
    paired = {f"{a}_vs_{b}": {k: compare(binary[a][k], binary[b][k], groups) for k in ("em", "joint", "abstained")}
              for a, b in CONTRASTS}
    return {"questions": len(cases), "outcomes": stats, "paired": paired}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("read", "reference"):
        p.add_argument("--" + name, type=Path, nargs=4, required=True)
    a = p.parse_args()
    fixture = unpack(a.manifest)
    cases = fixture["rows"]
    computes, ref = (sorted(((sha256(path.read_bytes()).hexdigest(), unpack(path)) for path in paths),
                            key=lambda item: item[1]["shard"]) for paths in (a.read, a.reference))
    metrics = {}
    common = ("reader_revision", "reader_weights_sha256", "labels_sha256", "arms")
    for shard, ((_, record), (src_hash, baseline)) in enumerate(zip(computes, ref, strict=True)):
        assert record["device"] == baseline["device"] == "NVIDIA L4"
        assert record["fixture_sha256"] == baseline["fixture_sha256"] == fixture["fixture_sha256"]
        assert (record["shard"], record["shards"]) == (baseline["shard"], baseline["shards"]) == (shard, 4)
        assert src_hash == record["reference_sha256"]
        assert set(record["arms"]) == set(ARMS) and all(record[k] == computes[0][1][k] for k in common)
        for q, row, original in zip(cases[shard::4], record["rows"], baseline["rows"], strict=True):
            assert q["id"] == row["id"] == original["id"]
            arms = row["arms"]
            assert set(arms) == set(ARMS)
            for arm, out in arms.items():
                assert len(out["turns"]) <= 20 and 0 < out["prompt_tokens"] <= 512
                flags = {q["source_session_ids"][i] for i, j in out["turns"] if j in q["author_turn_ids"][i]}
                assert out["marked_support"] == (set(q["gold_ids"]) <= flags)
                if arm in ("dense", "user"):
                    assert out["turns"] == original["turns_from_top5"][arm]
            for arm in ("dense", "oracle_marked"):
                original_arm, dated = arms[arm], arms[arm + "_dated"]
                assert all(original_arm[k] == dated[k] for k in ("turns", "retained_turn_tokens",
                                                                "original_turn_tokens", "clipped_turns"))
            assert set(i for i, _ in arms["marked_in_dense"]["turns"]) <= set(original["full"]["dense"][:5])
            metrics[q["id"]] = arms
    answerable = [q for q in cases if not q["id"].endswith("_abs")]
    scopes = {"answerable": answerable, "abstention": [q for q in cases if q["id"].endswith("_abs")]}
    for kind in sorted({q["kind"] for q in answerable}):
        scopes[kind] = [q for q in answerable if q["kind"] == kind]
    assert [len(scopes[k]) for k in ("answerable", "abstention")] == [470, 30]
    result = {"fixture_sha256": fixture["fixture_sha256"], **{k: computes[0][1][k] for k in common},
              "scopes": {k: summary(q, metrics) for k, q in scopes.items()}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
