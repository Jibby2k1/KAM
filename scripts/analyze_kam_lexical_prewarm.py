#!/usr/bin/env python3
"""Audit exact Tantivy rankings and crossed first-pass/warm timing."""

import argparse
from hashlib import sha256
import gzip
import json
from pathlib import Path
from random import Random

from analyze_kam_shared_serving import metrics


def summarize(base, comparison, ids):
    rows = [{"serving_timing_ms": {"reference": base[i]["ms"], "shared": comparison[i]["ms"]}} for i in ids]
    return metrics(rows)


def evaluate(ranks, fixture):
    rows = json.loads(gzip.decompress(fixture.read_bytes()))["rows"]["test"]
    ids = [q["id"] for shard in range(4) for q in rows[shard::4][4:132:4]]
    Random(382633).shuffle(ids)
    assert len(ids) == len(set(ids)) == 128
    assert [s["rank"] for s in ranks] == [0, 1] and len({s["node_sha256"] for s in ranks}) == 2
    fields = ("fixture_sha256", "index_sha256", "read_bytes", "read_files")
    common = {k: ranks[0][k] for k in fields}
    assert common["fixture_sha256"] == sha256(fixture.read_bytes()).hexdigest()
    assert all(all(s[k] == v for k, v in common.items()) and s["read_seconds"] > 0 for s in ranks)
    first, serial, parallel = {}, {}, {}
    local = []
    for s in ranks:
        r = s["rank"]
        assert [q["id"] for q in s["baseline"]] == ids[r::2]
        assert [q["id"] for q in s["prewarmed"]] == ids[1-r::2]
        rng = Random(82791 + r)
        for q in s["baseline"]:
            assert q["kind"] == "serial"
            first[q["id"]] = q
        for q in s["prewarmed"]:
            assert q["order"] == rng.sample(("serial", "two_threads"), k=2)
            serial[q["id"]] = q["routes"]["serial"]
            parallel[q["id"]] = q["routes"]["two_threads"]
            assert all(q["id"] == rec["id"] and key == rec["kind"] for key, rec in q["routes"].items())
        warm_ids = [q["id"] for q in s["prewarmed"]]
        local.append({"rank": r, "read_seconds": round(s["read_seconds"], 6),
                      "threading": summarize(serial, parallel, warm_ids)})
    assert first.keys() == serial.keys() == parallel.keys() == set(ids)
    for results in (first, serial, parallel):
        assert all(len(q["ordered"]) == len(set(q["ordered"])) == 128 and q["ms"] > 0 for q in results.values())
    return {**common, "by_host": local, "parity": {
                "prewarm_exact_order": sum(first[i]["ordered"] == serial[i]["ordered"] for i in ids),
                "threading_exact_order": sum(serial[i]["ordered"] == parallel[i]["ordered"] for i in ids),
                "prewarm_exact_set": sum(set(first[i]["ordered"]) == set(serial[i]["ordered"]) for i in ids),
                "threading_exact_set": sum(set(serial[i]["ordered"]) == set(parallel[i]["ordered"]) for i in ids)},
            "prewarm": summarize(first, serial, ids), "threading": summarize(serial, parallel, ids),
            "combined": summarize(first, parallel, ids)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", type=Path, nargs=2)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = evaluate([json.loads(f.read_text()) for f in a.profiles], a.fixture)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
