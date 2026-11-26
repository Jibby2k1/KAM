#!/usr/bin/env python3
"""Audit same-pool IIRC routing and bootstrap by source article."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

import numpy as np

from analyze_kam_shared_serving import unpack

ARMS = ("single_shot", "sequential", "pair", "pair_masked")
CONTRASTS = (("pair", "single_shot"), ("pair", "sequential"), ("pair", "pair_masked"),
             ("sequential", "single_shot"))


def paired(rows, treatment, control):
    by_source = {}
    deltas = []
    for row in rows:
        d = int(row["success"][treatment]) - int(row["success"][control])
        cluster = by_source.setdefault(row["source_id"], [0, 0])
        cluster[0] += d
        cluster[1] += 1
        deltas.append(d)
    clusters = np.array(list(by_source.values()), dtype=np.int64)
    samples = np.random.default_rng(3981).integers(len(clusters), size=(10000, len(clusters)))
    totals = clusters[samples].sum(axis=1)
    lo, hi = np.quantile(totals[:, 0] / totals[:, 1], (.025, .975))
    return {"wins": deltas.count(1), "losses": deltas.count(-1), "delta": round(sum(deltas) / len(rows), 8),
            "bootstrap95": [round(float(lo), 8), round(float(hi), 8)]}


def summarize(rows):
    return {"questions": len(rows), "source_articles": len({r["source_id"] for r in rows}),
            "arms": {arm: {"both_gold": sum(r["success"][arm] for r in rows),
                          "gold_articles_selected": sum(r["hits"][arm] for r in rows),
                          "warm_route_p50_ms": round(median(r["arms"][arm]["route_ms"] for r in rows), 6)}
                     for arm in ARMS},
            "paired": {a + "_vs_" + b: paired(rows, a, b) for a, b in CONTRASTS},
            "pair_or_sequential_oracle": sum(r["success"]["pair"] or r["success"]["sequential"] for r in rows)}


def evaluate(replay, manifest, source_audit):
    assert [s["shard"] for s in replay] == [0, 1, 2, 3]
    keys = ("fixture_sha256", "source_sha256", "config_sha256", "baseline_sha256", "assets", "device", "torch")
    provenance = {k: replay[0][k] for k in keys}
    assert provenance["device"] == "NVIDIA B200"
    assert all(all(s[k] == v for k, v in provenance.items()) and s["shards"] == 4 for s in replay)
    assert manifest["fixture_sha256"] == provenance["fixture_sha256"]
    assert manifest["source_sha256"] == provenance["source_sha256"]
    assert all(source_audit["sha256"][k] == v for k, v in provenance["source_sha256"].items())
    expected = manifest["rows"]
    assert len(expected) == source_audit["splits"]["dev"]["gold_link_count_answerable"]["2"] == 153
    assert len(expected) == len({r["id"] for r in expected})
    for shard in replay:
        actual = [{k: r[k] for k in ("id", "source_id", "pool_sha256", "candidates", "gold")}
                  for r in shard["rows"]]
        assert actual == expected[shard["shard"]::4]
    rows = sorted((r for s in replay for r in s["rows"]), key=lambda r: r["id"])
    assert len({r["source_id"] for r in rows}) == manifest["source_articles"]
    for r in rows:
        assert set(r["arms"]) == set(ARMS) and len(set(r["gold"])) == 2
        assert 2 < r["candidates"]
        assert all(0 <= i < r["candidates"] for i in r["gold"])
        r["hits"] = {}
        for arm in ARMS:
            chosen = r["arms"][arm]["selected"]
            assert len(chosen) == len(set(chosen)) == 2 and all(0 <= i < r["candidates"] for i in chosen)
            assert r["arms"][arm]["route_ms"] > 0
            r["hits"][arm] = len(set(chosen).intersection(r["gold"]))
        assert r["arms"]["single_shot"]["selected"][0] == r["arms"]["sequential"]["selected"][0]
        r["success"] = {arm: n == 2 for arm, n in r["hits"].items()}
    groups = {"all": rows, "title_link": [r for r in rows if r["gold_title_link"]],
              "no_title_link": [r for r in rows if not r["gold_title_link"]]}
    return {"provenance": provenance, "pool_widths": dict(sorted(Counter(r["candidates"] for r in rows).items())),
            "groups": {k: summarize(v) for k, v in groups.items() if v}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", type=Path)
    for name in ("manifest", "source-audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = evaluate(unpack(a.replay), json.loads(a.manifest.read_text()), json.loads(a.source_audit.read_text()))
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
