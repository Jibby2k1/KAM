#!/usr/bin/env python3
"""Audit fixed-cohort article fetch timings and reconcile staging metadata."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles


def evaluate(profiles, pools_path, reference_path):
    reference = json.loads(gzip.decompress(reference_path.read_bytes()))
    pools = json.loads(gzip.decompress(pools_path.read_bytes()))["rows"]
    case_sizes = {r["id"]: len(set(pools[r["id"]]) | set(r["corpus_hits"]["conditioned_union"]))
                  for s in reference for r in s["rows"]}
    ids = sorted(case_sizes)
    assert len(ids) == 1024 and set(ids) == pools.keys()
    assert [p["offset"] for p in profiles] == [0, 1]
    hashkeys = {"pools_sha256": sha256(pools_path.read_bytes()).hexdigest(),
                "reference_sha256": sha256(reference_path.read_bytes()).hexdigest()}
    methods = ("shared_first_touch", "shared_after_staging", "local_after_staging")
    shards = []
    for p in profiles:
        assert all(p[k] == v for k, v in hashkeys.items())
        assert p["database_sha256"] == profiles[0]["database_sha256"]
        assert p["database_bytes"] == profiles[0]["database_bytes"]
        assert p["stride"] == 8 and not p["same_filesystem_device"]
        assert [r["id"] for r in p["rows"]] == ids[p["offset"]::8]
        assert len(p["rows"]) == p["exact_text_parity"] == 128
        assert all(r["articles"] == case_sizes[r["id"]] for r in p["rows"])
        computed = {}
        for label in methods:
            vals = [r[label] for r in p["rows"]]
            assert all(v > 0 for v in vals)
            computed[label] = {"p50_ms": round(median(vals), 6), "p95_ms": round(quantiles(vals, n=20)[18], 6)}
        assert computed == p["summary"]
        shards.append({"offset": p["offset"], "copy_seconds": p["copy_seconds"], "requests": len(p["rows"]),
                       "stats": computed})
    return {**hashkeys, "database_bytes": profiles[0]["database_bytes"],
            "database_sha256": profiles[0]["database_sha256"], "requests_with_equal_article_text": 256,
            "profiles": shards}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", nargs=2, type=Path)
    for name in ("pools", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    data = [json.loads(x.read_text()) for x in a.profiles]
    a.output.write_text(json.dumps(evaluate(data, a.pools, a.reference), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
