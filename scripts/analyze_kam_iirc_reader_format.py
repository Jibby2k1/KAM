#!/usr/bin/env python3
"""Compare IIRC answer prompting while holding evidence and decoder fixed."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from analyze_kam_iirc_pair import paired
from analyze_kam_shared_serving import unpack

EVIDENCE = ("article_leads", "annotated_spans")
POLICIES = ("generic", "universal", "question_type", "oracle_type")
TYPES = ("span", "binary", "value")


def compare(rows, evidence):
    selected = [{**r, "success": {p: r["labels"][evidence + ":" + p]["exact"] for p in POLICIES}} for r in rows]
    result = {"questions": len(selected), "source_articles": len({r["source_id"] for r in selected}),
              "exact": {p: sum(r["success"][p] for r in selected) for p in POLICIES},
              "paired": {p + "_vs_generic": paired(selected, p, "generic") for p in POLICIES[1:]},
              "output_forms": {p: dict(sorted(Counter(r["labels"][evidence + ":" + p]["output_type"]
                                                     for r in selected).items())) for p in POLICIES}}
    return result


def audit(shards, reference):
    archive = unpack(reference)
    assert [s["shard"] for s in shards] == list(range(4))
    pro_keys = ("fixture_sha256", "span_counts_sha256", "source_sha256", "excluded_multispan", "assets", "gpu")
    provenance = {k: archive[0][k] for k in pro_keys}
    assert all(all(s[k] == v for k, v in provenance.items()) and s["shards"] == 4 for s in shards + archive)
    assert all(tuple(s["format_policies"]) == POLICIES for s in shards)
    truths = {r["id"]: r for shard in archive for r in shard["rows"]}
    rows = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == 134 and {r["id"] for r in rows} == truths.keys()
    checks = {e: Counter() for e in EVIDENCE}
    trimmed = dict.fromkeys(EVIDENCE, 0)
    type_confusion = {t: dict.fromkeys(TYPES, 0) for t in TYPES}
    for s in shards:
        assert s["rows"] == rows[s["shard"]::4]
    fields = ("prompt_tokens", "exact", "output_sha256")
    for r in rows:
        expected = truths[r["id"]]
        assert all(r[k] == expected[k] for k in ("id", "source_id", "answer_type"))
        assert set(r["labels"]) == {e + ":" + p for e in EVIDENCE for p in POLICIES}
        for v in r["labels"].values():
            assert type(v["exact"]) is bool and 0 < v["prompt_tokens"] <= 512
            assert re.fullmatch("[0-9a-f]{64}", v["output_sha256"])
            assert v["output_type"] in ("binary", "value", "other")
        type_confusion[r["answer_type"]][r["inferred_type"]] += 1
        assert set(r["matched_budget_trim"]) == set(EVIDENCE)
        for e in EVIDENCE:
            assert type(r["matched_budget_trim"][e]) is bool
            trimmed[e] += r["matched_budget_trim"][e]
            for k in fields:
                checks[e][k] += r["labels"][e + ":generic"][k] == expected["labels"][e][k]
    groups = {"all": rows} | {t: [r for r in rows if r["answer_type"] == t] for t in TYPES}
    return {"provenance": provenance, "reference_sha256": sha256(reference.read_bytes()).hexdigest(),
            "generic_parity": checks, "shared_budget_trims": trimmed, "inferred_type_confusion": type_confusion,
            "results": {e: {name: compare(group, e) for name, group in groups.items()} for e in EVIDENCE}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = audit(unpack(a.replay), a.reference)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
