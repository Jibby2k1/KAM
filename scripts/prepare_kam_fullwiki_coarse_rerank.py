#!/usr/bin/env python3
"""Seal approximate corpus rankings for the unchanged HotpotQA reranker."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_shared_serving import unpack


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search", nargs=4, required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=("probe_64", "probe_256", "full_rescore"))
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    reference = sorted(unpack(a.reference), key=lambda x: x["shard"])
    metadata = ("fixture_sha256", "index_sha256", "pools_sha256", "fitted_sha256")
    primary = {key: reference[0][key] for key in metadata}
    assert all(all(c[k] == primary[k] for k in metadata) for c in reference)
    ranks = sorted(map(unpack, a.search), key=lambda x: x["shard"])
    assert [x["shard"] for x in ranks] == [x["shard"] for x in reference] == list(range(4))
    rows = {}
    for ranked, past in zip(ranks, reference, strict=True):
        assert all(ranked[k] == primary[k] for k in metadata[:-1])
        assert ranked["archive_sha256"] == sha256(a.reference.read_bytes()).hexdigest()
        original = {q["id"]: q for q in past["rows"]}
        assert set(original) == {q["id"] for q in ranked["rows"]}
        for r in ranked["rows"]:
            q = original[r["id"]]
            ids = r["methods"][a.mode]["ranks"]
            assert q["first"] not in ids and len(ids) == len(set(ids)) == 64
            assert set(r["methods"]["exact"]["ranks"]) == set(q["corpus_hits"]["conditioned_union"])
            rows[r["id"]] = {"first": q["first"], "unconditioned_union": q["corpus_hits"]["unconditioned_union"],
                             "conditioned_union": ids}
    data = primary | {"search_method": a.mode, "partition_sha256": ranks[0]["partition_sha256"],
                      "source_search_sha256": [sha256(p.read_bytes()).hexdigest() for p in a.search], "rows": rows}
    a.output.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
