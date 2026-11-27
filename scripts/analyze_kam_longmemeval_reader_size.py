#!/usr/bin/env python3
"""Separate fixed-prompt reader scale, evidence visibility and abstention."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from kam_flan_reader_assets import LARGE_REVISION, LARGE_WEIGHT
from run_kam_longmemeval_m_reader import SCALE_VARIANTS

SIZES = ("base", "large")
METRICS = ("answer_em", "joint_em", "abstention")


def score(cases, outcomes):
    group = shared_evidence_clusters([{"gold": q["gold_ids"] or [q["id"]]} for q in cases])
    result = {}
    for name in SCALE_VARIANTS:
        rows = [outcomes[q["id"]][name] for q in cases]
        flags, counts = {}, {}
        for size in SIZES:
            values = {k: [] for k in METRICS}
            f1 = []
            for row in rows:
                main = row["readers"][size]
                flags_here = (main["em"], main["em"] and row["marked_support"], main["abstained"])
                for key, value in zip(METRICS, flags_here, strict=True):
                    values[key].append(value)
                f1.append(main["f1"])
            flags[size] = values
            counts[size] = {k: sum(v) for k, v in values.items()} | {"mean_f1": mean(f1)}
        result[name] = {"counts": counts, "paired": {k: compare(flags["large"][k], flags["base"][k], group) for k in METRICS},
                        "marked_support": sum(r["marked_support"] for r in rows),
                        "median_prompt_tokens": median(r["prompt_tokens"] for r in rows)}
    return {"questions": len(cases), "clusters": len(group), "results": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "output"):
        p.add_argument("--" + arg, required=True, type=Path)
    for arg in ("trials", "baseline"):
        p.add_argument("--" + arg, required=True, type=Path, nargs=4)
    a = p.parse_args()
    fixture = unpack(a.manifest)
    cases = fixture["rows"]
    baseline, experiment = [sorted(map(unpack, paths), key=lambda r: r["shard"]) for paths in (a.baseline, a.trials)]
    outcomes, same = {}, Counter()
    costs = {size: [] for size in SIZES}
    for i, (old, run) in enumerate(zip(baseline, experiment, strict=True)):
        assert (old["shard"], run["shard"], old["shards"], run["shards"]) == (i, i, 4, 4)
        assert old["device"] == run["device"] == "NVIDIA L4"
        assert old["fixture_sha256"] == run["fixture_sha256"] == fixture["fixture_sha256"]
        assert all(old[k] == run[k] for k in ("reference_sha256", "labels_sha256"))
        assert run["weights"]["large"] == {"revision": LARGE_REVISION, "sha256": LARGE_WEIGHT}
        assert run["weights"]["base"] == {"revision": old["reader_revision"], "sha256": old["reader_weights_sha256"]}
        assert run["arms"] == list(SCALE_VARIANTS)
        for size in SIZES:
            costs[size].append(run["prediction_seconds"][size])
        for q, old_row, new_row in zip(cases[i::4], old["rows"], run["rows"], strict=True):
            id_ = q["id"]
            assert id_ == old_row["id"] == new_row["id"] and id_ not in outcomes
            for name, (_, cap, _, _) in SCALE_VARIANTS.items():
                r = new_row["arms"][name]
                assert r["prompt_tokens"] <= cap and set(r["readers"]) == set(SIZES)
                if name in old_row["arms"]:
                    assert r | r["readers"]["base"] | {"readers": None} == old_row["arms"][name] | {"readers": None}
                    same[name] += 1
            outcomes[id_] = new_row["arms"]
    assert len(outcomes) == len(cases) == 500
    eligible = [q for q in cases if not q["id"].endswith("_abs")]
    scopes = {"answerable": eligible, "abstention": [q for q in cases if q["id"].endswith("_abs")]}
    scopes.update({key: [q for q in eligible if q["kind"] == key] for key in sorted({q["kind"] for q in eligible})})
    payload = {"fixture_sha256": fixture["fixture_sha256"], "identical_base_replays": same,
               "shard_prediction_seconds": costs, "median_shard_prediction_seconds": {k: median(v) for k, v in costs.items()},
               "scopes": {name: score(rows, outcomes) for name, rows in scopes.items()}}
    a.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
