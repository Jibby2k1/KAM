#!/usr/bin/env python3
"""Audit hard endpoint selection, paired exact match and matched wall time."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from analyze_kam_native_musique import compare
from analyze_kam_shared_serving import unpack


def groups(rows, field):
    result = defaultdict(list)
    for i, r in enumerate(rows):
        result[r[field]].append(i)
    return list(result.values())


def aggregate(rows, a, b):
    family = groups(rows, "family")
    pairs = groups(rows, "pair")
    questions = groups(rows, "question")
    assert all(len(p) == 4 for p in pairs) and all(len(q) == 2 for q in questions)
    return {"correct": sum(a), "all_presentations_correct": sum(all(a[i] for i in p) for p in pairs),
            "order_invariant_correct": sum(all(a[i] for i in q) for q in questions),
            "both_orders_agree": sum(a[q[0]] == a[q[1]] for q in questions),
            "contrast": compare(a, b, family)}


def outcomes(rows):
    dense, answer = {}, {}
    get = lambda key: [r["dense_endpoints"][key] for r in rows]
    for mode in rows[0]["dense_endpoints"]:
        dense[mode] = aggregate(rows, get(mode), get("endpoints"))
    for size in rows[0]["readers"]:
        tests = {}
        for category in ("readers", "routed"):
            for policy in rows[0][category][size]:
                values = [r[category][size][policy] for r in rows]
                baseline = [r["readers"][size]["true_prefix"]["exact"] for r in rows]
                exact = [v["exact"] for v in values]
                tests[policy] = {"answers": aggregate(rows, exact, baseline), "foil_answers": sum(v["foil_answer"] for v in values)}
                if category == "routed":
                    joint = [r["exact"] and r["support_exact"] for r in values]
                    tests[policy].update(joint=aggregate(rows, joint, baseline), support=sum(v["support_exact"] for v in values),
                                         median_physical_tokens=median(v["physical_tokens"] for v in values),
                                         median_visible_tokens=median(v["visible_tokens"] for v in values))
                else:
                    tests[policy]["median_physical_tokens"] = median(r["tokens"][policy] for r in rows)
        answer[size] = tests
    return {"presentations": len(rows), "questions": len(rows)//2, "pairs": len(rows)//4,
            "families": len(groups(rows, "family")), "dense": dense, "reader": answer}


def timing_summary(shards):
    by_policy = defaultdict(list)
    for shard in shards:
        for row in shard.get("timing", []):
            if not row["warmup"]:
                width = row["width"]
                assert len(row["cases"]) == width
                assert all(len(p["exact"]) == width and p["wall_ms"] > 0 for p in row["phases"].values())
                by_policy[(row["reader"], width)].append(row["phases"])
    results = {}
    for (reader, width), runs in by_policy.items():
        old, route = ([r[label]["wall_ms"] for r in runs] for label in ("both_with_prefix", "route_then_answer"))
        results[f"{reader}/{width}"] = {
            "batches": len(runs), "examples": len(runs)*width, "median_both_ms": median(old), "median_routed_ms": median(route),
            "median_paired_saving_ms": median(x-y for x, y in zip(old, route, strict=True)),
            "routed_faster_batches": sum(x > y for x, y in zip(old, route, strict=True)),
            "certainty_replay_matches": {label: sum(r[label]["prior_matches"] for r in runs) for label in runs[0]},
        }
    return results


def audit(fixture_path, result_paths, previous_paths):
    fixture = unpack(fixture_path)
    fingerprint = sha256(fixture_path.read_bytes()).hexdigest()
    results = list(map(unpack, result_paths))
    previous = {unpack(p)["shard"]: p for p in previous_paths}
    assert len(results) == 4 and {s["shard"] for s in results} == set(previous) == set(range(4))
    records, rejected, kept = [], [], []
    for shard in results:
        index = shard["shard"]
        assert shard["device"] == "NVIDIA L4" and shard["fixture_sha256"] == fingerprint
        assert shard["source_sha256"] == fixture["source_sha256"] and shard["shards"] == 4
        ref = previous[index]
        assert shard["reference_sha256"] == sha256(ref.read_bytes()).hexdigest()
        old = unpack(ref)
        assert old["models"] == shard["models"] and old["model_revisions"] == shard["model_revisions"]
        assert old["rejected"] == shard["rejected"]
        assert [{k: v for k, v in row.items() if k != "routed"} for row in shard["rows"]] == old["rows"]
        expected = [(p["id"], q["id"], reverse) for p in fixture["pairs"][index::4] if p["id"] not in old["rejected"]
                    for q in p["questions"] for reverse in (False, True)]
        assert expected == [(r["pair"], r["question"], r["reverse"]) for r in shard["rows"]]
        for row in shard["rows"]:
            assert row["tokens"]["true_prefix"] == row["tokens"]["cross_prefix"]
            for sizes in row["routed"].values():
                assert sizes["true_prefix_route_masked"]["physical_tokens"] == row["tokens"]["true_prefix"]
                assert sizes["true_prefix_route"]["support_exact"] == row["dense_endpoints"]["true_prefix"]
                assert sizes["query_route"]["support_exact"] == row["dense_endpoints"]["endpoints"]
                assert sizes["cross_prefix_route"]["support_exact"] == row["dense_endpoints"]["cross_prefix"]
                assert all(not (s["exact"] and s["foil_answer"]) for s in sizes.values())
        records.extend(shard["rows"])
        rejected.extend(shard["rejected"])
        kept.extend(row["pair"] for row in shard["rows"])
    info = {k: v for k, v in fixture.items() if k != "pairs"}
    info["fixture_sha256"] = fingerprint
    manifest = {**info, "rejected_pairs": rejected, "pairs": [{"id": p["id"], "hops": p["hops"], "family": p["family"],
                   "questions": [q["id"] for q in p["questions"]], "evidence": [[d["id"] for d in q["evidence"]] for q in p["questions"]]}
                  for p in fixture["pairs"]]}
    summary = {**info, "accepted_pairs": len(set(kept)), "rejected_pairs": len(rejected),
               "quality": {str(h): outcomes([r for r in records if r["hops"] == h]) for h in sorted({r["hops"] for r in records})},
               "latency": timing_summary(results)}
    return summary, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--measurements", type=Path, nargs=4, required=True)
    p.add_argument("--references", type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    summary, manifest = audit(a.fixture, a.measurements, a.references)
    for target, rows in ((a.output, summary), (a.manifest, manifest)):
        target.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
