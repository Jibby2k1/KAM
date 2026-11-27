#!/usr/bin/env python3
"""Compare early independently bound routes at the same MuSiQue evidence budget."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

ARMS = ("top1", "first_span_same", "first_span_second")
BUDGETS = {2: 32, 3: 22, 4: 16}
NEW = {"first_alternative_used", "first_alternative_em", "alternate_searches", "alternate_target_ranks"}
OLD_ALTERNATE = {"alternative_search", "alternate_final_rank"}


def align(manifest, reference, measurements, fit):
    stamps = {v["weights_sha256"]: v["seed"] for v in fit}
    assert len(stamps) == 3 and all(r["arm"] == "trained" for r in fit)
    wanted = {(h, i) for h in stamps for i in range(4)}
    old = {(p["span_checkpoint_sha256"], p["shard"]): p for p in reference}
    new = {(p["span_checkpoint_sha256"], p["shard"]): p for p in measurements}
    assert len(old) == len(new) == len(wanted) == 12 and set(old) == set(new) == wanted
    paired, times = {v: {} for v in stamps.values()}, {v: {} for v in stamps.values()}
    checked = Counter()
    provenance = ("fixture_sha256", "corpus_sha256", "models", "templates_sha256", "scoring_labels_sha256",
                  "baseline_sha256", "branch_labels_sha256")
    for key, block in new.items():
        h, i = key
        seed = stamps[h]
        prior = old[key]
        assert block["shards"] == prior["shards"] == 4 and block["device"] == prior["device"] == "NVIDIA L4"
        assert all(block[k] == prior[k] for k in provenance)
        cases = manifest["rows"][i::4]
        for case, r, xx in zip(cases, block["rows"], prior["rows"], strict=True):
            assert case["id"] == r["id"] == xx["id"]
            assert set(r["arms"]) == set(ARMS)
            a = r["arms"]["top1"]
            b = xx["arms"]["top1"]
            assert {k: v for k, v in a.items() if k not in NEW} == {k: v for k, v in b.items() if k not in OLD_ALTERNATE}
            assert not a["first_alternative_used"] and a["alternate_searches"] == 0
            checked[seed] += 1
            p = 64 + BUDGETS[case["hops"]]
            for alternative in ARMS[1:]:
                d = r["arms"][alternative]
                assert all(a[k] == d[k] for k in a if k not in NEW | {"pool", "fixed_evidence_ids", "fixed_answer_em"})
                assert d["pool"][:p] == a["pool"][:p] and d["fixed_evidence_ids"][0] == a["fixed_evidence_ids"][0]
                assert len(d["pool"]) == len(set(d["pool"])) == 128
                assert len(d["fixed_evidence_ids"]) == len(set(d["fixed_evidence_ids"])) == 4
                assert set(d["fixed_evidence_ids"]) <= set(d["pool"])
                assert d["alternate_target_ranks"][0] is None
                assert 0 <= d["alternate_searches"] < case["hops"]
                if not d["first_alternative_used"]:
                    assert a == d
            assert r["id"] not in paired[seed]
            paired[seed][r["id"]] = r["arms"]
        for h, values in block["branch_seconds"].items():
            sums = times[seed].setdefault(h, Counter())
            assert set(values) == set(ARMS)
            sums.update(values)
    assert set(checked.values()) == {len(manifest["rows"])}
    return paired, times, checked


def flags(rec, gold):
    support = gold <= set(rec["fixed_evidence_ids"])
    return {"complete_pool": gold <= set(rec["pool"]), "selected_support": support,
            "exact_answer": bool(rec["fixed_answer_em"]), "joint": support and bool(rec["fixed_answer_em"])}


def gold_outcomes(manifest, rows, measurements):
    out = {}
    for block in rows:
        assert block["fixture_sha256"] == measurements[0]["fixture_sha256"] == manifest["fixture_sha256"]
        assert block["corpus_sha256"] == measurements[0]["corpus_sha256"]
        assert block["label_sha256"] == measurements[0]["branch_labels_sha256"]
        assert block["reader_revision"] == measurements[0]["models"]["google/flan-t5-base"]
        for q, r in zip(manifest["rows"][block["shard"]::block["shards"]], block["rows"], strict=True):
            arm = r["arms"]["gold_only"]
            assert q["id"] == r["id"] and set(arm["evidence_ids"]) == set(q["gold"]) and q["id"] not in out
            out[q["id"]] = bool(arm["em"])
    assert len(out) == len(manifest["rows"])
    return out


def analyze(manifest, reference, measurements, fit, gold_controls):
    paired, processing, checked = align(manifest, reference, measurements, fit)
    golden = gold_outcomes(manifest, gold_controls, measurements)
    out = {}
    for depth in COHORTS:
        cases = [q for q in manifest["rows"] if q["hops"] == depth]
        groups = shared_evidence_clusters(cases)
        output = {"cases": len(cases), "clusters": len(groups), "seeds": {}}
        for seed, all_rows in sorted(paired.items()):
            source = [all_rows[q["id"]] for q in cases]
            gold = [set(q["gold"]) for q in cases]
            reference_flags = [flags(r["top1"], g) for r, g in zip(source, gold, strict=True)]
            result = {}
            oracle = [golden[q["id"]] for q in cases]
            oracle_ref = [a and b["complete_pool"] for a, b in zip(oracle, reference_flags, strict=True)]
            for policy in ARMS:
                found = [flags(r[policy], g) for r, g in zip(source, gold, strict=True)]
                read = [r[policy] for r in source]
                numbers = {metric: sum(r[metric] for r in found) for metric in found[0]}
                uncertainty = {m: compare([r[m] for r in found], [o[m] for o in reference_flags], groups) for m in numbers}
                if depth == 4:
                    recover = [a and b["complete_pool"] for a, b in zip(oracle, found, strict=True)]
                    numbers["gold_oracle_joint"] = sum(recover)
                    uncertainty["gold_oracle_joint"] = compare(recover, oracle_ref, groups)
                result[policy] = {"counts": numbers, "vs_control": uncertainty,
                                  "different_first_answers": sum(x["first_alternative_used"] for x in read),
                                  "different_first_answers_correct": sum(x["first_alternative_em"] for x in read),
                                  "additional_corpus_queries": sum(x["alternate_searches"] for x in read),
                                  "total_measured_shard_seconds": processing[seed][str(depth)][policy]}
            output["seeds"][seed] = result
        out[str(depth)] = output
    return {"primary_reference_checks": checked, "scopes": out}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--reference", type=Path, nargs=12, required=True)
    p.add_argument("--measurements", type=Path, nargs=12, required=True)
    p.add_argument("--fit", type=Path, nargs=3, required=True)
    p.add_argument("--gold-reader", type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fitted = [json.loads(f.read_bytes()) for f in a.fit]
    result = analyze(unpack(a.fixture), list(map(unpack, a.reference)), list(map(unpack, a.measurements)), fitted, list(map(unpack, a.gold_reader)))
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
