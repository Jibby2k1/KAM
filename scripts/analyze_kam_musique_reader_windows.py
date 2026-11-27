#!/usr/bin/env python3
"""Reconcile fixed-evidence reader outcomes with the frozen MuSiQue fixture."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack


def audit(path, results):
    content = path.read_bytes()
    fixture = json.loads(content)
    stamp = sha256(content).hexdigest()
    assert len(results) == 4 and {block["shard"] for block in results} == set(range(4))
    reference = results[0]
    for block in results:
        assert block["device"] == "NVIDIA L4" and block["shards"] == 4
        assert block["fixture_sha256"] == stamp and block["corpus_sha256"] == fixture["corpus_sha256"]
        assert block["models"] == reference["models"]
        expected = [q["id"] for q in fixture["rows"][block["shard"]::4] if q["hops"] == 4]
        assert expected == [r["id"] for r in block["rows"]]
        assert block["exact_archive_replays"] == 2 * len(expected)
    records = {r["id"]: r["panels"] for block in results for r in block["rows"]}
    cases = [q for q in fixture["rows"] if q["hops"] == 4]
    assert len(records) == len(cases) == 160
    clusters = shared_evidence_clusters(cases)
    result = {}
    for panel in next(iter(records.values())):
        rows = [records[q["id"]][panel] for q in cases]
        truth = [set(q["gold"]) for q in cases]
        modes = reference["times"]["base"]
        data = {}
        for mode in modes:
            evidence = [r["details"][mode]["evidence"] for r in rows]
            for r, group in zip(rows, evidence, strict=True):
                found = r["details"][mode]
                assert len(found["prompt_sha256"]) == 64 and found["tokens"] <= 1024
                assert set(group) <= set(r["details"]["archived_96"]["evidence"])
                assert len(group) == (1 if "_only_" in mode else 4)
            coverage = [expected <= set(group) for expected, group in zip(truth, evidence, strict=True)]
            if panel == "gold_paragraphs" and "_only_" not in mode:
                assert all(coverage)
            data[mode] = {"complete_evidence": sum(coverage),
                          "median_prompt_tokens": median(r["details"][mode]["tokens"] for r in rows), "models": {}}
            for size in reference["models"]:
                flags = [r["judged"][size][mode] for r in rows]
                archive = [r["judged"][size]["archived_96"] for r in rows]
                all_wide = [r["judged"][size]["prefix_192"] for r in rows]
                joint = [a and b for a, b in zip(flags, coverage, strict=True)]
                if size == "base" and mode == "archived_96":
                    assert flags == [r["archived_exact"] for r in rows]
                data[mode]["models"][size] = {"answer_exact": sum(flags), "joint_exact": sum(joint),
                                               "vs_size_matched_archive": compare(flags, archive, clusters),
                                               "vs_size_matched_wide": compare(flags, all_wide, clusters)}
            a, b = ([r["judged"][size][mode] for r in rows] for size in ("large", "base"))
            data[mode]["large_vs_base"] = compare(a, b, clusters)
        result[panel] = data
    return {"fixture_sha256": stamp, "questions": len(cases), "shared_evidence_clusters": len(clusters),
            "archive_replays": sum(b["exact_archive_replays"] for b in results), "panels": result,
            "summed_generation_seconds": {size: {mode: sum(b["times"][size][mode] for b in results)
                                                   for mode in reference["times"][size]}
                                          for size in reference["times"]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--measurements", type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    results = audit(a.fixture, list(map(unpack, a.measurements)))
    a.output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
