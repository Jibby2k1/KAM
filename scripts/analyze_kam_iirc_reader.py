#!/usr/bin/env python3
"""Audit fixed-article IIRC answers with paired intervals by source article."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median

from analyze_kam_iirc_pair import paired
from analyze_kam_shared_serving import unpack

ARMS = ("question", "source_lead", "article_leads", "linked_oracle_windows", "annotated_spans")
CONTRASTS = (("source_lead", "question"), ("article_leads", "source_lead"),
             ("linked_oracle_windows", "article_leads"), ("annotated_spans", "article_leads"),
             ("annotated_spans", "linked_oracle_windows"))


def summarize(rows):
    return {"questions": len(rows), "source_articles": len({r["source_id"] for r in rows}),
            "exact": {arm: sum(r["success"][arm] for r in rows) for arm in ARMS},
            "prompt_tokens": {arm: {"p50": median(r["labels"][arm]["prompt_tokens"] for r in rows),
                                    "max": max(r["labels"][arm]["prompt_tokens"] for r in rows)} for arm in ARMS},
            "paired": {a + "_vs_" + b: paired(rows, a, b) for a, b in CONTRASTS}}


def evaluate(shards, visibility):
    source_file = visibility.read_bytes()
    counts = unpack(visibility)
    assert [s["shard"] for s in shards] == list(range(4))
    assert all(s["shards"] == 4 for s in shards)
    keys = ("fixture_sha256", "source_sha256", "span_counts_sha256", "excluded_multispan", "gpu", "assets")
    provenance = {key: shards[0][key] for key in keys}
    assert all(all(s[k] == v for k, v in provenance.items()) for s in shards)
    assert provenance["source_sha256"] == counts["source_sha256"]
    assert provenance["span_counts_sha256"] == sha256(source_file).hexdigest()
    aligned = [r for r in counts["rows"]
               if not any(d["alignment"].get("unresolved", 0) for d in (r["source"], *r["linked"]))]
    assert len(aligned) == 139
    excluded = set(provenance["excluded_multispan"])
    assert len(excluded) == 5 and excluded <= {r["id"] for r in aligned if r["answer_type"] == "span"}
    expected = [r for r in aligned if r["id"] not in excluded]
    rows = sorted((r for s in shards for r in s["rows"]), key=lambda r: r["id"])
    assert len(rows) == 134
    for s in shards:
        assert s["rows"] == rows[s["shard"]::4]
    id_keys = ("id", "source_id", "answer_type")
    assert [[r[k] for k in id_keys] for r in rows] == [[r[k] for k in id_keys] for r in expected]
    same_output = Counter()
    binary_outputs = {arm: Counter() for arm in ARMS}
    literal_binary = {sha256((form + suffix).encode()).hexdigest(): value
                      for value in ("yes", "no") for form in (value, value.title(), value.upper())
                      for suffix in ("", ".")}
    for row in rows:
        labels = row["labels"]
        assert set(labels) == set(ARMS)
        row["success"] = {a: labels[a]["exact"] for a in ARMS}
        for a in ARMS:
            assert type(labels[a]["exact"]) is bool
            assert 0 < labels[a]["prompt_tokens"] <= 512
            assert re.fullmatch("[0-9a-f]{64}", labels[a]["output_sha256"])
            if row["answer_type"] == "binary":
                binary_outputs[a][literal_binary.get(labels[a]["output_sha256"], "other")] += 1
        for a, b in CONTRASTS:
            same_output[a + "_vs_" + b] += labels[a]["output_sha256"] == labels[b]["output_sha256"]
    types = ("span", "binary", "value")
    return {"provenance": provenance, "identical_output": dict(same_output),
            "literal_binary_outputs": {arm: dict(sorted(counts.items())) for arm, counts in binary_outputs.items()},
            "all": summarize(rows),
            "by_answer_type": {k: summarize([r for r in rows if r["answer_type"] == k]) for k in types}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", type=Path)
    p.add_argument("--visibility", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = evaluate(unpack(a.replay), a.visibility)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
