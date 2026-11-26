#!/usr/bin/env python3
"""Freeze a title-disjoint natural routing pilot from the public dev split."""

import argparse
import json
from pathlib import Path
import unicodedata

from audit_kam_natural_data import digest, stable


def title_key(title):
    return " ".join(unicodedata.normalize("NFKC", title).casefold().replace("_", " ").split())


def prepare(source, feasibility):
    published = json.loads(source.read_text())
    by_id = {row["_id"]: row for row in published}

    def titles(case):
        return {title_key(title) for title, _ in case["context"]}

    def eligible(case):
        support = {title_key(title) for title, _ in case["supporting_facts"]}
        return (case["type"] == "bridge" and len(titles(case)) == 10
                and len(support) == 2 and support <= titles(case))

    pilot_ids = {item["id"] for item in json.loads(feasibility.read_text())["sample"]}
    used = {title for id_ in pilot_ids for title in titles(by_id[id_])}
    eligible_rows = sorted(filter(eligible, published), key=lambda case: stable("router:" + case["_id"]))
    transfer = [case for case in eligible_rows if not (titles(case) & used)][:256]
    heldout_titles = set().union(*(titles(case) for case in transfer))
    train = [case for case in eligible_rows if not (titles(case) & (used | heldout_titles))][:1024]
    assert len(train) == 1024 and len(transfer) == 256
    assert not ({case["_id"] for case in train + transfer} & pilot_ids)
    assert not (set().union(*(titles(case) for case in train)) & heldout_titles)
    return {
        "scope": "development only; transfer candidate titles and feasibility ids are absent from training",
        "source_sha256": digest(source), "feasibility_sha256": digest(feasibility),
        "training": [case["_id"] for case in train], "transfer": [case["_id"] for case in transfer],
        "training_unique_candidate_titles": len(set().union(*(titles(case) for case in train))),
        "transfer_unique_candidate_titles": len(heldout_titles),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("feasibility", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(prepare(a.source, a.feasibility), indent=2) + "\n")


if __name__ == "__main__":
    main()
