#!/usr/bin/env python3
"""Compare atomic dependency tokens with text references on frozen MuSiQue plans."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZipFile

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack
from audit_kam_natural_data import digest
from prepare_kam_native_musique import MEMBERS

METHODS = ("text", "atomic", "atomic_backward")
REFERENCE = re.compile(r"#(\d+)\b")


def refs(steps):
    return [sorted({int(x) for x in REFERENCE.findall(s)}) for s in steps]


def authored(source, fixture):
    assert digest(source) == fixture["source_sha256"]
    ids = {q["id"] for q in fixture["rows"]}
    output = {}
    seen = sha256()
    with ZipFile(source) as archive, archive.open(MEMBERS["dev"]) as file:
        for line in file:
            seen.update(line)
            row = json.loads(line)
            if row["id"] in ids:
                assert row["answerable"]
                output[row["id"]] = refs([item["question"] for item in row["question_decomposition"]])
    assert seen.hexdigest() == fixture["member_sha256"]["dev"] and output.keys() == ids
    return output


def panel(cases, observed):
    stats = {f"{seed}": {mode: Counter() for mode in METHODS} for seed in range(3)}
    metrics = ("all_support", "right_step_count", "valid", "author_dependency_graph")
    flags = {f"{seed}": {mode: {m: [] for m in metrics} for mode in METHODS} for seed in range(3)}
    for q in cases:
        for i, evidence in enumerate(observed[q["id"]]):
            for mode, data in evidence.items():
                stats[str(i)][mode].update(data)
                for k in metrics:
                    flags[str(i)][mode][k].append(data[k])
    groups = shared_evidence_clusters(cases)
    pairs = {}
    for i, record in flags.items():
        pairs[i] = {f"{a}_vs_{b}": {k: compare(record[a][k], record[b][k], groups) for k in metrics}
                    for a, b in (("atomic", "text"), ("atomic_backward", "text"), ("atomic_backward", "atomic"))}
    return {"questions": len(cases), "counts": stats, "comparisons": pairs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    for name, n in (("legacy-fit", 3), ("atomic-fit", 3), ("legacy-plans", 3), ("atomic-plans", 3),
                    ("backward-plans", 3), ("legacy-ranks", 12), ("new-ranks", 24), ("baseline", 4)):
        p.add_argument("--" + name, required=True, nargs=n, type=Path)
    a = p.parse_args()
    fixture = unpack(a.manifest)
    cases = fixture["rows"]
    gold_ref = authored(a.source, fixture)
    assert len(cases) == 576
    source_ranks = {digest(f): unpack(f) for f in a.baseline}
    ranks = {}
    for ranking in map(unpack, (*a.legacy_ranks, *a.new_ranks)):
        ranks.setdefault(ranking["generation_sha256"], []).append(ranking)
    observed = {q["id"]: [dict() for _ in range(3)] for q in cases}
    seeds = []
    for i, (old_file, new_file, *plans) in enumerate(zip(a.legacy_fit, a.atomic_fit, a.legacy_plans, a.atomic_plans,
                                                         a.backward_plans, strict=True)):
        old, new = (json.loads(f.read_text()) for f in (old_file, new_file))
        match = ("seed", "data_sha256", "fixture_sha256", "revision", "steps", "seed_per_hop", "device")
        assert all(old[k] == new[k] for k in match) and new["fixture_sha256"] == fixture["fixture_sha256"]
        assert new["binding_tokens"] and new["device"] == "NVIDIA L4"
        seeds.append(new["seed"])
        for mode, pf in zip(METHODS, plans, strict=True):
            generated = unpack(pf)
            assert generated["fixture_sha256"] == fixture["fixture_sha256"] and generated["device"] == new["device"]
            assert generated.get("binding_tokens", False) == (mode != "text")
            assert generated.get("enforce_binding_order", False) == (mode == "atomic_backward")
            choices = generated["rows"]
            shards = sorted(ranks[digest(pf)], key=lambda r: r["shard"])
            assert [r["shard"] for r in shards] == list(range(4))
            for shard, results in enumerate(shards):
                old_rank = source_ranks[results["baseline_sha256"]]
                assert results["fixture_sha256"] == old_rank["fixture_sha256"] == fixture["fixture_sha256"]
                assert results["corpus_sha256"] == old_rank["corpus_sha256"] == fixture["corpus_sha256"]
                assert results["weights_sha256"] == generated["weights_sha256"] and old_rank["shard"] == shard
                for q, plan, row, baseline in zip(cases[shard::4], choices[shard::4], results["rows"], old_rank["rows"], strict=True):
                    id_ = q["id"]
                    assert id_ == plan["id"] == row["id"] == baseline["id"]
                    reference = gold_ref[id_]
                    graph = refs(plan["steps"])
                    gold = set(q["gold"])
                    assert len(row["pool"]) == len(set(row["pool"])) == 128
                    assert row["fallback"] is not plan["valid"] and row["predicted_steps"] == len(graph)
                    if row["fallback"]:
                        assert row["pool"] == baseline["hybrid"]
                    observed[id_][i][mode] = {"all_support": gold <= set(row["pool"]), "right_step_count": len(graph) == q["hops"],
                                             "valid": plan["valid"], "author_dependency_graph": graph == reference,
                                             "hybrid_support": gold <= set(baseline["hybrid"])}
    scopes = {"all": cases} | {str(d): [q for q in cases if q["hops"] == d] for d in (2, 3, 4)}
    report = {"fixture_sha256": fixture["fixture_sha256"], "seeds": seeds,
              "scopes": {k: panel(v, observed) for k, v in scopes.items()}}
    a.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
