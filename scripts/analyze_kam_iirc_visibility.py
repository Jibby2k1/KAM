#!/usr/bin/env python3
"""Summarize fixed-window IIRC visibility and paired article-level uncertainty."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_iirc_pair import paired
from analyze_kam_shared_serving import unpack

POLICIES = ("lead", "bm25_window", "oracle_window")


def analyze(data, manifest_path):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert data["manifest_sha256"] == sha256(manifest_bytes).hexdigest()
    assert data["source_sha256"] == manifest["source_sha256"]
    assert (data["source_words"], data["window_words"], data["stride_words"]) == (100, 160, 80)
    rows = data["rows"]
    assert [(r["id"], r["source_id"]) for r in rows] == [(r["id"], r["source_id"]) for r in manifest["rows"]]
    assert len(rows) == 153 and len({r["id"] for r in rows}) == 153
    alignment = {"source": Counter(), "linked": Counter()}
    aligned, missing = [], Counter()
    for row in rows:
        source, linked = row["source"], row["linked"]
        assert len(linked) == 2
        docs = [source, *linked]
        for i, doc in enumerate(docs):
            assert doc["spans"] > 0
            assert sum(doc["alignment"].values()) == doc["spans"]
            assert all(0 <= n <= doc["spans"] - doc["alignment"].get("unresolved", 0)
                       for n in doc["visible"].values())
            alignment["linked" if i else "source"].update(doc["alignment"])
        assert set(source["visible"]) == {"source"} and not source["chosen_window"]
        for doc in linked:
            assert set(doc["visible"]) == set(doc["chosen_window"]) == set(POLICIES)
            assert doc["chosen_window"]["lead"] == 0 and all(i >= 0 for i in doc["chosen_window"].values())
            assert doc["visible"]["oracle_window"] >= max(doc["visible"]["bm25_window"], doc["visible"]["lead"])
        if any(doc["alignment"].get("unresolved", 0) for doc in docs):
            missing[row["answer_type"]] += 1
        else:
            aligned.append(row)

    def measure(metric):
        counts = []
        for row in aligned:
            source, linked = row["source"], row["linked"]
            success = {}
            for policy in POLICIES:
                one_each = all(doc["visible"][policy] > 0 for doc in linked)
                linked_all = all(doc["visible"][policy] == doc["spans"] for doc in linked)
                source_all = source["visible"]["source"] == source["spans"]
                success[policy] = {"one_each_linked": one_each, "all_linked": linked_all,
                                   "all_source_and_linked": source_all and linked_all}[metric]
            counts.append({"source_id": row["source_id"], "success": success})
        return {"questions": len(counts), "totals": {p: sum(r["success"][p] for r in counts) for p in POLICIES},
                "bm25_vs_lead": paired(counts, "bm25_window", "lead"),
                "oracle_vs_bm25": paired(counts, "oracle_window", "bm25_window")}

    return {"manifest_sha256": data["manifest_sha256"], "source_sha256": data["source_sha256"],
            "all": len(rows), "fully_aligned": len(aligned), "excluded_by_answer_type": missing,
            "aligned_source_articles": len({r["source_id"] for r in aligned}), "alignment": alignment,
            "fully_visible_source": sum(r["source"]["visible"]["source"] == r["source"]["spans"] for r in aligned),
            "linked_spans_total": sum(d["spans"] for r in aligned for d in r["linked"]),
            "visible_linked_spans": {p: sum(d["visible"][p] for r in aligned for d in r["linked"]) for p in POLICIES},
            "metrics": {m: measure(m) for m in ("one_each_linked", "all_linked", "all_source_and_linked")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = analyze(unpack(a.replay), a.manifest)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
