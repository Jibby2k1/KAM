#!/usr/bin/env python3
"""Measure first-hop BM25 recall as native MuSiQue distractors enter the pool."""
import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

from audit_kam_natural_data import digest, rank, stable
from kam_math_kl_protocol import write_json
from prepare_kam_native_musique import MEMBERS, canonical_paragraph

PROTOCOL = "kam-musique-native-lexical-v1"
WIDTHS = (2, 8, 20)


def audit(source, panel, manifest_path, live_summary, gpu_fixture=None):
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(live_summary.read_text())
    assert manifest["source_sha256"] == digest(source) and manifest["panel_sha256"] == digest(panel)
    assert summary["manifest_sha256"] == digest(manifest_path)
    omitted = set(summary["excluded_pairs"])
    pairs = [row for row in json.loads(panel.read_text()) if row["id"] not in omitted]
    targets = {q["id"]: (i, pair) for pair in pairs for i, q in enumerate(pair["questions"])}
    assert len(targets) == sum(len(pair["questions"]) for pair in pairs)
    records, full_pools = {}, {}
    with ZipFile(source) as archive:
        for split, member in MEMBERS.items():
            with archive.open(member) as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["id"] not in targets:
                        continue
                    index, pair = targets[row["id"]]
                    question = pair["questions"][index]
                    assert pair["split"] == split and question["question"] == row["question"]
                    support = [p["evidence"][0] for p in pair["questions"]]
                    excluded = {p["id"] for q in pair["questions"] for p in q["evidence"]}
                    added = {}
                    for paragraph in row["paragraphs"]:
                        identity = stable("\x1f".join(canonical_paragraph(paragraph)))
                        if not paragraph["is_supporting"] and identity not in excluded:
                            added[identity] = paragraph["title"] + ". " + paragraph["paragraph_text"]
                    order = sorted(added, key=lambda ident: stable(PROTOCOL + row["id"] + ident))
                    if gpu_fixture:
                        full_pools[row["id"]] = [p["title"] + ". " + p["body"] for p in support] + [added[k] for k in order]
                    ranks = {}
                    for width in WIDTHS:
                        extras = order[:width - 2]
                        passages = [p["title"] + ". " + p["body"] for p in support] + [added[k] for k in extras]
                        forward = rank(row["question"], passages, limit=len(passages))
                        backward = rank(row["question"], passages[::-1], limit=len(passages))
                        ranks[width] = [forward.index(index) + 1, backward.index(len(passages) - 1 - index) + 1]
                    records[row["id"]] = {"id": row["id"], "family": pair["family"], "split": split,
                                           "distractors": order, "first_hop_ranks": ranks}
    assert records.keys() == targets.keys()
    if gpu_fixture:
        write_json(gpu_fixture, {"pairs": pairs, "first_hop_pools": full_pools})
    aggregate = {}
    for split in MEMBERS:
        cohort = [r for r in records.values() if r["split"] == split]
        aggregate[split] = {"n": len(cohort), "distractor_counts": Counter(len(r["distractors"]) for r in cohort),
                            "recall_both_orders": {
            width: {f"top_{k}": sum(max(r["first_hop_ranks"][width]) <= k for r in cohort)
                    for k in (1, 2, 4) if k <= width} for width in WIDTHS
        }}
    return {"protocol": PROTOCOL, "manifest_sha256": digest(manifest_path), "summary_sha256": digest(live_summary),
            "aggregate": aggregate, "records": [records[ident] for ident in targets]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("source", "panel", "manifest", "live_summary", "output"):
        parser.add_argument(item, type=Path)
    parser.add_argument("--gpu-fixture", type=Path)
    args = parser.parse_args()
    write_json(args.output, audit(args.source, args.panel, args.manifest, args.live_summary, args.gpu_fixture))
