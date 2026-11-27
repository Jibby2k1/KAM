#!/usr/bin/env python3
"""Audit two open MuSiQue hops at a fixed 128-candidate budget."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_musique_endpoint_pairs import groups
from analyze_kam_native_musique import compare
from analyze_kam_shared_serving import unpack


def audit(measurements, references):
    previous = {unpack(p)["shard"]: p for p in references}
    runs = list(map(unpack, measurements))
    assert len(runs) == 4 and {r["shard"] for r in runs} == set(previous) == set(range(4))
    keys = ("fixture_sha256", "corpus_sha256", "source_sha256", "device", "models")
    assert all(all(r[k] == runs[0][k] for k in keys) for r in runs) and runs[0]["device"] == "NVIDIA L4"
    cases = []
    for r in runs:
        path = previous[r["shard"]]
        assert r["prior_sha256"] == sha256(path.read_bytes()).hexdigest()
        archived = unpack(path)
        assert archived["fixture_sha256"] == r["fixture_sha256"] and archived["model_revisions"] == r["models"]
        ref = [row for row in archived["rows"] if row["hops"] == 2 and not row["reverse"]]
        for q, previous_row in zip(r["rows"],ref,strict=True):
            assert all(q[k] == previous_row[k] for k in ("question", "pair", "family"))
            assert len(set(q["annotated_route"])) == 2
            for family in ("dense", "hybrid", "bm25"):
                a,b,c = [q["systems"][f"{family}/{k}"] for k in ("direct", "same_pool", "conditioned_search")]
                assert a["pool"] == b["pool"] and a["pool"][:64] == c["pool"][:64]
                assert a["selected"] == a["pool"][:2] and len({x["selected"][0] for x in (a,b,c)}) == 1
                for x in (a,b,c):
                    assert x["contains_both"] == (set(q["annotated_route"]) <= set(x["pool"]))
                    assert len(set(x["selected"])) == 2 and set(x["selected"]) <= set(x["pool"])
                    assert len(set(x["pool"])) == len(x["pool"]) == 128
            cases.append(q)
    assert len(cases) == len({x["question"] for x in cases}) == 168
    clusters = groups(cases, "family")
    results = {}
    for name in cases[0]["systems"]:
        family, _ = name.split("/",1)
        outcomes = {k: [row["systems"][k] for row in cases] for k in (name,family+"/direct",family+"/same_pool")}
        labels = [q["annotated_route"] for q in cases]
        variants = {}
        for k, rows in outcomes.items():
            strict = [set(v["selected"]) == set(g) for v,g in zip(rows,labels,strict=True)]
            variants[k] = {"coverage": [r["contains_both"] for r in rows], "selected": strict,
                           "ordered_both": [r["selected"] == g for r,g in zip(rows,labels,strict=True)],
                           "both_text_exact": [r["exact_answer_both"] for r in rows],
                           "terminal_exact": [r["exact_answer_second_only"] for r in rows],
                           "both_text_joint": [r["exact_answer_both"] and hit for r,hit in zip(rows,strict,strict=True)],
                           "terminal_joint": [r["exact_answer_second_only"] and hit for r,hit in zip(rows,strict,strict=True)]}
        scores = variants[name]
        measured = outcomes[name]
        kept_anchor = [r["selected"][0] in g for r, g in zip(measured, labels, strict=True)]
        pools_full = scores["coverage"]
        headroom = {
            "incomplete_pool": sum(not a for a in pools_full),
            "complete_pool_unannotated_anchor": sum(a and not b for a, b in zip(pools_full, kept_anchor, strict=True)),
            "complete_pool_annotated_anchor_wrong_second": sum(a and b and not pair
                for a, b, pair in zip(pools_full, kept_anchor, scores["selected"], strict=True)),
            "selected_pair_wrong_terminal_readout": sum(a and not b for a, b in zip(scores["selected"], scores["terminal_exact"], strict=True)),
            "joint_terminal_success": sum(scores["terminal_joint"]),
        }
        assert sum(headroom.values()) == len(cases)
        results[name] = {"counts": {m: sum(v) for m,v in scores.items()},
                         "headroom": headroom,
                         "vs_direct": {m: compare(v,variants[family+"/direct"][m],clusters) for m,v in scores.items()},
                         "vs_same_pool": {m: compare(v,variants[family+"/same_pool"][m],clusters) for m,v in scores.items()},
                         "terminal_vs_both_joint": compare(scores["terminal_joint"],scores["both_text_joint"],clusters)}
    return {k:runs[0][k] for k in keys} | {"questions":len(cases), "families":len(clusters), "candidate_budget":128, "results":results}


def audit_rerank(measurements, references):
    prior = {unpack(p)["shard"]: p for p in references}
    current = list(map(unpack, measurements))
    assert len(current) == 4 and {r["shard"] for r in current} == set(prior) == set(range(4))
    rows = []
    for r in current:
        assert r["device"] == "NVIDIA L4"
        path = prior[r["shard"]]
        initial = unpack(path)
        assert r["open_reference_sha256"] == sha256(path.read_bytes()).hexdigest()
        assert all(r[k] == initial[k] for k in ("fixture_sha256", "corpus_sha256", "source_sha256", "models"))
        for a, b in zip(r["rows"], initial["rows"], strict=True):
            assert all(a[k] == b[k] for k in ("question", "pair", "family", "annotated_route"))
            baseline = b["systems"]["dense/conditioned_search"]
            replay = a["results"]["conditioned"]
            assert a["pool"] == baseline["pool"] and len(set(a["pool"])) == 128
            assert replay["selected"] == baseline["selected"]
            assert replay["answer_exact"] == baseline["exact_answer_second_only"]
            assert all(v["selected"][0] == a["anchor"] and len(set(v["selected"])) == 2
                       and set(v["selected"]) <= set(a["pool"]) for v in a["results"].values())
            rows.append(a)
    assert len(rows) == len({q["question"] for q in rows}) == 168
    family = groups(rows, "family")
    outcomes = {}
    for tag in rows[0]["results"]:
        records = [r["results"][tag] for r in rows]
        pair = [set(x["selected"]) == set(r["annotated_route"]) for r, x in zip(rows, records, strict=True)]
        joint = [p and x["answer_exact"] for p, x in zip(pair, records, strict=True)]
        baseline = [set(r["results"]["conditioned"]["selected"]) == set(r["annotated_route"])
                    and r["results"]["conditioned"]["answer_exact"] for r in rows]
        outcomes[tag] = {"correct_pair": sum(pair), "joint": sum(joint),
                         "selected_title_mentioned": sum(x["mentioned_second"] for x in records),
                         "vs_conditioned_joint": compare(joint, baseline, family)}
    return {"questions": len(rows), "families": len(family), "baseline_replayed": len(rows), "outcomes": outcomes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, nargs=4, required=True)
    p.add_argument("--references", type=Path, nargs=4, required=True)
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summarize = audit_rerank if a.rerank else audit
    a.output.write_text(json.dumps(summarize(a.measurements, a.references), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
