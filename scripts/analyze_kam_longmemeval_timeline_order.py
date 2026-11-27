#!/usr/bin/env python3
"""Compare dates and chronology with frozen LongMemEval turns and token content."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from run_kam_longmemeval_m_reader import TIMELINE_VARIANTS, prompt_order

SIZES = ("base", "large")
METRICS = ("answer", "joint", "abstained")
PAIRS = (("dense_rank_dated", "dense_rank_undated"), ("dense_ascending", "dense_rank_dated"),
         ("dense_descending", "dense_rank_dated"), ("dense_shuffled", "dense_rank_dated"),
         ("dense_ascending", "dense_shuffled"), ("dense_descending", "dense_shuffled"),
         ("oracle_rank_dated", "oracle_rank_undated"), ("oracle_ascending", "oracle_rank_dated"),
         ("oracle_descending", "oracle_rank_dated"))
UNDATED = {"dense_rank_undated": "dense_1024_window", "oracle_rank_undated": "oracle_1024_prefix"}


def assess(cases, replay):
    groups = shared_evidence_clusters([{"gold": q["gold_ids"] or [q["id"]]} for q in cases])
    output = {}
    for size in SIZES:
        arms = {}
        for arm in TIMELINE_VARIANTS:
            rows = [replay[q["id"]][arm] for q in cases]
            a = [v["readers"][size]["em"] for v in rows]
            arms[arm] = {"answer": a, "joint": [hit and r["marked_support"] for hit, r in zip(a, rows, strict=True)],
                         "abstained": [v["readers"][size]["abstained"] for v in rows]}
        output[size] = {"counts": {name: {k: sum(x) for k, x in vals.items()} for name, vals in arms.items()},
                        "paired": {f"{a}_vs_{b}": {k: compare(arms[a][k], arms[b][k], groups) for k in METRICS}
                                   for a, b in PAIRS}}
    return {"questions": len(cases), "clusters": len(groups), "readers": output}


def eligible(q):
    known = {x for x, date in zip(q["source_session_ids"], q["dates"], strict=True) if date <= q["time"]}
    return set(q["gold_ids"]) <= known


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "output"):
        p.add_argument("--" + option, required=True, type=Path)
    for option in ("reference", "timeline"):
        p.add_argument("--" + option, required=True, type=Path, nargs=4)
    a = p.parse_args()
    manifest = unpack(a.manifest)
    cases = manifest["rows"]
    control, trials = [sorted(map(unpack, arr), key=lambda r: r["shard"]) for arr in (a.reference, a.timeline)]
    records, parity, changes = {}, Counter(), Counter()
    for shard, (base, test) in enumerate(zip(control, trials, strict=True)):
        assert all(p["shard"] == shard and p["shards"] == 4 for p in (base, test))
        assert all(base[k] == test[k] for k in ("fixture_sha256", "reference_sha256", "labels_sha256", "weights", "device"))
        assert test["fixture_sha256"] == manifest["fixture_sha256"] and test["device"] == "NVIDIA L4"
        assert test["arms"] == list(TIMELINE_VARIANTS)
        for q, old, row in zip(cases[shard::4], base["rows"], test["rows"], strict=True):
            id_ = q["id"]
            assert id_ == old["id"] == row["id"] and id_ not in records
            arms = row["arms"]
            for m, (policy, dated, method) in TIMELINE_VARIANTS.items():
                current, ref = arms[m], arms[policy.replace("oracle_marked", "oracle") + "_rank_undated"]
                assert current["dated"] == dated and current["prompt_tokens"] <= 1024
                keys = ("turns", "clipped_turns", "body_sha256", "retained_turn_tokens", "marked_support", "shifted_turns")
                assert all(current[k] == ref[k] for k in keys)
                assert current["prompt_order"] == [current["turns"][i] for i in prompt_order(q, current["turns"], method)]
                changes[m] += current["prompt_order"] != current["turns"]
                if m in UNDATED:
                    prev = old["arms"][UNDATED[m]]
                    assert all(current[k] == prev[k] for k in ("prompt_tokens", "turns", "retained_turn_tokens", "marked_support"))
                    for size in SIZES:
                        parity[f"{m}/{size}"] += current["readers"][size] == prev["readers"][size]
            records[id_] = arms
    assert len(records) == len(cases) == 500
    valid = [q for q in cases if not q["id"].endswith("_abs")]
    temporal = [q for q in valid if q["kind"] == "temporal-reasoning"]
    scopes = {"answerable": valid, "abstention": [q for q in cases if q["id"].endswith("_abs")],
              "temporal": temporal, "temporal_known_as_of": [q for q in temporal if eligible(q)],
              "temporal_later_evidence": [q for q in temporal if not eligible(q)]}
    result = {"fixture_sha256": manifest["fixture_sha256"], "undated_score_replay": parity, "order_changes": changes,
              "scopes": {name: assess(panel, records) for name, panel in scopes.items() if panel}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
