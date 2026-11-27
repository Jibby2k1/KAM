#!/usr/bin/env python3
"""Measure MuSiQue chain recovery and binding errors on fixed author templates."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

METHODS = ("top1", "top3", "gold_paragraph")


def assess(cases, outputs, previous):
    labels = {q["id"]: set(q["gold"]) for q in cases}
    hits = {"annotated_only": [], "resolved_answers": []}
    totals = {name: Counter() for name in hits}
    for id_, truth in labels.items():
        for arm in hits:
            n = len(truth.intersection(previous[id_][arm]))
            hits[arm].append(n == len(truth))
            totals[arm].update({"full_support": n == len(truth), "support_paragraphs": n})
    controls = ("annotated_only", "resolved_answers")
    stages = {a: [{} for _ in range(max(q["hops"] for q in cases))] for a in METHODS}
    for arm in METHODS:
        hits[arm] = []
        totals[arm] = Counter()
        for q in cases:
            id_ = q["id"]
            expected = labels[id_]
            o = outputs[id_]["arms"][arm]
            assert len(o["pool"]) == len(set(o["pool"])) == 128
            found = len(expected.intersection(o["pool"]))
            hit = found == q["hops"]
            hits[arm].append(hit)
            totals[arm].update({"full_support": hit, "support_paragraphs": found,
                                "all_intermediates_em": all(o["answer_em"][:-1]),
                                "last_subquestion_em": bool(o["answer_em"][-1]),
                                "all_and_last_em": hit and bool(o["answer_em"][-1]),
                                "unfilled_references": o["unfilled_references"]})
            paths = zip(o["target_ranks"], o["answer_em"], o["gold_in_reader"], strict=True)
            for i, (r, em, context) in enumerate(paths):
                stage = stages[arm][i]
                for label, value in (("n", 1), ("gold_top1", r == 1), ("gold_top16", r is not None and r <= 16),
                                     ("gold_in_context", context), ("answer_em", em),
                                     ("answer_em_with_gold", context and em)):
                    stage[label] = stage.get(label, 0) + int(value)
    groups = shared_evidence_clusters(cases)
    comparisons = [(a, b) for a in METHODS for b in controls] + [("top3", "top1"), ("gold_paragraph", "top3")]
    return {"questions": len(cases), "metrics": totals, "stages": stages,
            "paired_full_support": {f"{a}_vs_{b}": compare(hits[a], hits[b], groups) for a, b in comparisons}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    for arm in ("baseline", "recursive", "bindings"):
        p.add_argument("--" + arm, required=True, nargs=4, type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = unpack(a.manifest)
    baseline = {sha256(p.read_bytes()).hexdigest(): unpack(p) for p in a.baseline}
    prior = {c["shard"]: c for c in map(unpack, a.recursive)}
    chunks = sorted(map(unpack, a.bindings), key=lambda c: c["shard"])
    assert len({c["templates_sha256"] for c in chunks}) == len({c["scoring_labels_sha256"] for c in chunks}) == 1
    assert [s["shard"] for s in chunks] == list(range(4))
    items, orig = {}, {}
    for part in chunks:
        key = part["shard"]
        first = baseline[part["baseline_sha256"]]
        previous = prior[key]
        assert part["fixture_sha256"] == previous["fixture_sha256"] == manifest["fixture_sha256"]
        assert part["corpus_sha256"] == previous["corpus_sha256"] == manifest["corpus_sha256"]
        assert part["baseline_sha256"] == previous["baseline_sha256"]
        assert part["device"] == "NVIDIA L4" and first["shard"] == key and part["shards"] == 4
        assert all(part["models"][k] == v for k, v in previous["models"].items())
        for item, base, old, q in zip(part["rows"], first["rows"], previous["rows"], manifest["rows"][key::4], strict=True):
            id_ = item["id"]
            assert id_ == base["id"] == old["id"] == q["id"] and set(item["arms"]) == set(METHODS)
            for name, arm in item["arms"].items():
                assert arm["pool"][:64] == base["dense"][:64] and arm["id"] == id_
                assert all(0 <= i < manifest["documents"] for i in arm["pool"])
                assert all(len(r) == q["hops"] for r in
                           (arm["target_ranks"], arm["answer_em"], arm["gold_in_reader"], arm["reader_context_ids"]))
                if name == "gold_paragraph":
                    assert all(arm["gold_in_reader"])
            items[id_], orig[id_] = item, old
    cs = manifest["rows"]
    scopes = {"all": cs, **{str(n): [q for q in cs if q["hops"] == n] for n in COHORTS}}
    out = {"fixture_sha256": manifest["fixture_sha256"], "templates_sha256": chunks[0]["templates_sha256"],
           "scopes": {k: assess(v, items, orig) for k, v in scopes.items()}}
    a.output.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
