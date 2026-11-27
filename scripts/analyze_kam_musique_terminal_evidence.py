#!/usr/bin/env python3
"""Audit terminal-passage and matched answer edits on the same four-hop fixture."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack


def audit(fixture, paths, references):
    content = fixture.read_bytes()
    stamp = sha256(content).hexdigest()
    rows = json.loads(content)["rows"]
    matched = {}
    former = {unpack(ref)["shard"]: ref for ref in references}
    assert set(former) == set(range(4))
    checkpoints = None
    for path in paths:
        block = unpack(path)
        index = block["shard"]
        ref = former[index]
        assert block["fixture_sha256"] == stamp and block["device"] == "NVIDIA L4" and block["shards"] == 4
        assert block["window_reference_sha256"] == sha256(ref.read_bytes()).hexdigest()
        old = unpack(ref)
        assert old["models"] == block["models"]
        checkpoints = checkpoints or block["models"]
        assert block["models"] == checkpoints
        expected = [x for x in rows[index::4] if x["hops"] == 4]
        panel = (x["panels"]["gold_paragraphs"]["judged"] for x in old["rows"])
        for case, result, gold in zip(expected, block["rows"], panel, strict=True):
            assert case["id"] == result["id"] and result["id"] not in matched
            for size, variants in result["judged"].items():
                assert variants["all"]["original_exact"] == gold[size]["prefix_192"]
                assert variants["terminal_full"]["original_exact"] == gold[size]["last_only_192"]
                assert all(not (x["original_exact"] and x["donor_exact"]) for x in variants.values())
                assert ("terminal_replaced" in variants) == result["rewrite_eligible"]
                assert ("all_terminal_replaced" in variants) == result["rewrite_eligible"]
            matched[result["id"]] = result
    cases = [q for q in rows if q["hops"] == 4]
    assert len(cases) == len(matched) == 160
    display = [matched[q["id"]] for q in cases]
    clusters = shared_evidence_clusters(cases)
    eligible = [r for r in display if r["rewrite_eligible"]]
    locations = Counter()
    for r in display:
        locations.update(r["surface_answer_occurrence"])
    results = {}
    for size in checkpoints:
        outcomes = {}
        for name in display[0]["judged"][size]:
            if name.endswith("_replaced"):
                continue
            flags = [r["judged"][size][name]["original_exact"] for r in display]
            baseline = [r["judged"][size]["all"]["original_exact"] for r in display]
            outcomes[name] = {"n": len(flags), "original_exact": sum(flags), "vs_all": compare(flags, baseline, clusters),
                              "exact_with_terminal_body_occurrence": sum(f and q["surface_answer_occurrence"]["body"]
                                                                         for f, q in zip(flags, display, strict=True))}
        for before, after in (("all", "all_terminal_replaced"), ("terminal_full", "terminal_replaced")):
            a = [r["judged"][size][before]["original_exact"] for r in eligible]
            b = [r["judged"][size][after] for r in eligible]
            outcomes[after] = {"n": len(eligible), "before_original_exact": sum(a),
                               "after_original_exact": sum(r["original_exact"] for r in b),
                               "after_donor_exact": sum(r["donor_exact"] for r in b),
                               "followed_change_given_original_exact": sum(p and r["donor_exact"] for p, r in zip(a, b, strict=True))}
        results[size] = outcomes
    return {"fixture_sha256": stamp, "models": checkpoints, "questions": len(cases), "matched_rewrites": len(eligible),
            "shared_evidence_clusters": len(clusters), "literal_answer_locations": locations, "outcomes": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--measurements", type=Path, nargs=4, required=True)
    p.add_argument("--references", type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(audit(a.fixture, a.measurements, a.references), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
