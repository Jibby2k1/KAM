#!/usr/bin/env python3
"""Verify crossed end-to-end request timings around a shared-index prewarm."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import median, quantiles

from analyze_kam_shared_serving import metrics


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def stats(rows):
    values = [r["request_ms"] for r in rows]
    assert all(t > 0 for t in values)
    return {"p50_ms": round(median(values), 6), "p95_ms": round(quantiles(values, n=20)[18], 6)}


def evaluate(profiles, fixture_path, reference_path):
    ref = unpack(reference_path)
    fixture = gzip.decompress(fixture_path.read_bytes())
    ids = sorted(q["id"] for q in json.loads(fixture)["rows"]["test"])
    assert len(ids) == len(set(ids)) == 1024
    expected = {r["id"]: r for s in ref for r in s["rows"]}
    assert set(ids) == expected.keys()
    selected = Random(20261127).sample(ids, 256)
    assert [p["index_profile_rank"] for p in profiles] == [0, 1]
    assert len({p["node_sha256"] for p in profiles}) == 2
    assert {p["gpu"] for p in profiles} == {"NVIDIA B200"}
    assert len({p["database_bytes"] for p in profiles}) == 1
    keys = ("fixture_sha256", "pools_sha256", "config_sha256", "index_sha256", "fitted_sha256", "heads_sha256",
            "ablations_sha256", "baseline_sha256", "models", "indexed_documents", "guard_columns", "split")
    frozen = {k: ref[0][k] for k in keys}
    assert frozen["fixture_sha256"] == sha256(fixture).hexdigest()
    assert all(all(s[k] == v for k, v in frozen.items()) for s in (*ref, *profiles))
    before, after, by_node = {}, {}, []
    for p in profiles:
        rank = p["index_profile_rank"]
        cold, warm = ([r for r in p["rows"] if r["index_warmed"] == b] for b in (False, True))
        assert [q["id"] for q in cold] == selected[rank::2]
        assert [q["id"] for q in warm] == selected[1 - rank::2]
        assert p["prewarm_seconds"] > 0
        before.update((r["id"], r) for r in cold)
        after.update((r["id"], r) for r in warm)
        by_node.append({"rank": rank, "prewarm_seconds": p["prewarm_seconds"], "baseline": stats(cold),
                        "prewarmed": stats(warm)})
    assert set(selected) == before.keys() == after.keys()

    fields = ("pool_hash", "first", "corpus_hits", "arms", "features")
    parity = {k: sum(before[i][k] == after[i][k] for i in selected) for k in fields}
    for i in selected:
        assert before[i]["pool_hash"] == after[i]["pool_hash"] == expected[i]["pool_hash"]
    archived = {label: sum(all(row[i]["arms"][k][f] == expected[i]["arms"][k][f]
                              for k in row[i]["arms"] for f in ("pair", "exact")) for i in selected)
                for label, row in (("baseline", before), ("prewarmed", after))}
    pairs = [{"serving_timing_ms": {"reference": before[i]["request_ms"], "shared": after[i]["request_ms"]}}
             for i in selected]
    t = metrics(pairs)
    timings = {"baseline": t.pop("reference"), "prewarmed": t.pop("shared"), **t}
    timings["prewarmed_faster"] = timings.pop("shared_faster")
    return {**frozen, "reference_sha256": sha256(reference_path.read_bytes()).hexdigest(), "gpu": "NVIDIA B200",
            "database_bytes": profiles[0]["database_bytes"], "parity": parity,
            "batched_archive_evidence_and_exactness": archived, "timing": timings, "by_node": by_node}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", nargs=2, type=Path)
    for name in ("fixture", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = evaluate([unpack(x) for x in a.profiles], a.fixture, a.reference)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
