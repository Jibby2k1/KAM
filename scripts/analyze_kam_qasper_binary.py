#!/usr/bin/env python3
"""Measure paired ground-truth effects of evidence on yes/no QASPER answers."""

import argparse
import json
from pathlib import Path

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from run_kam_qasper_binary import ARMS

FIELDS = ("support", "forced_correct", "free_correct", "joint")


def summaries(case_rows, outcomes):
    groups = shared_evidence_clusters([{"gold": [q["paper"]]} for q in case_rows])
    flags = {m: {k: [] for k in FIELDS} for m in ARMS}
    for q in case_rows:
        for name, out in outcomes[q["id"]].items():
            selected = set(out["slots"])
            support = any(set(option) <= selected for option in q["gold_sets"])
            forced = out["yes"] == q["answer"]
            free = out["free_generation"] == q["answer"]
            for key, value in zip(FIELDS, (support, forced, free, support and forced), strict=True):
                flags[name][key].append(value)
    counts = {m: {k: sum(flags) for k, flags in data.items()} for m, data in flags.items()}
    paired = {f"{m}_vs_{b}": {k: compare(flags[m][k], flags[b][k], groups) for k in FIELDS}
              for m, b in (("gold", "same_paper_sham"), ("gold", "no_evidence"),
                           ("paper_dense", "no_evidence"), ("paper_dense", "gold"))}
    return {"questions": len(case_rows), "paper_clusters": len(groups), "counts": counts, "paired": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--measurements", type=Path, nargs=2, required=True)
    a = parser.parse_args()
    fixture = unpack(a.manifest)
    raw = fixture["rows"]
    trials = sorted(map(unpack, a.measurements), key=lambda q: q["shard"])
    nums, tags = {}, ("fixture_sha256", "reader_weights_sha256", "reader_revision", "codes")
    for i, trial in enumerate(trials):
        assert (trial["shard"], trial["shards"]) == (i, 2) and trial["device"] == "NVIDIA L4"
        assert trial["fixture_sha256"] == fixture["fixture_sha256"] and all(trial[k] == trials[0][k] for k in tags)
        for q, row in zip(raw[i::2], trial["rows"], strict=True):
            assert row["id"] == q["id"] and set(row["arms"]) == set(ARMS)
            for k, output in row["arms"].items():
                assert output["yes"] == (output["yes_minus_no_logit"] > 0)
                assert 0 < output["tokens"] <= 512 and len(output["slots"]) == len(set(output["slots"])) <= 2
                if k == "gold":
                    assert output["slots"] == q["gold"]
                if k == "same_paper_sham":
                    assert output["slots"] == q["sham"]
            nums[q["id"]] = row["arms"]
    assert len(nums) == len(raw)
    groups = {"all": raw, "yes": [r for r in raw if r["answer"]], "no": [r for r in raw if not r["answer"]]}
    result = {"fixture_sha256": fixture["fixture_sha256"],
              "scopes": {name: summaries(group, nums) for name, group in groups.items()}}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
