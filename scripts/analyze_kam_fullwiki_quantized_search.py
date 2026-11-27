#!/usr/bin/env python3
"""Measure index-space savings, candidate parity and coverage after INT8 shortlisting."""

import argparse
from collections import Counter
from hashlib import sha256
import gzip
import json
from pathlib import Path
from statistics import median

import numpy as np

from analyze_kam_native_musique import compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

MODES = ("fp16_rescore", "int8_rescore")
LIMITS = ("96", "192", "256")


def run(raw_fixture, measurement, device):
    stamp = sha256(raw_fixture).hexdigest()
    fixture = json.loads(raw_fixture)["rows"]["test"]
    by_id = {q["id"]: q for q in fixture}
    meta = {k: measurement[0][k] for k in ("index_sha256", "candidate_docs", "resident_bytes", "models")}
    found, timings = {}, []
    for shard, panel in enumerate(measurement):
        assert (panel["shard"], panel["shards"]) == (shard, 4)
        assert panel["device"] == device and panel["fixture_sha256"] == stamp
        assert all(panel[k] == v for k, v in meta.items()) and len(panel["rows"]) == 32
        this_shard = {r["id"] for r in fixture[shard::4]}
        for batch in panel["rows"]:
            ids = batch["ids"]
            assert len(ids) == len(set(ids)) == 8 and set(ids) <= this_shard
            assert not set(ids).intersection(found)
            assert set(batch["timing_ms"]) == set(batch["ranks"]) == set(MODES)
            timings.append(tuple(batch["timing_ms"][m] for m in MODES))
            f = batch["ranks"]["fp16_rescore"]["96"]
            new = batch["ranks"]["int8_rescore"]
            assert len(f) == 8 and set(new) == set(LIMITS) and all(len(v) == 8 for v in new.values())
            for i, id_ in enumerate(ids):
                outputs = {d: new[d][i] for d in LIMITS}
                assert all(len(v) == len(set(v)) == 64 for v in (f[i], *outputs.values()))
                found[id_] = {"fp16": f[i], **outputs}
    assert len(fixture) == len(found) == 1024
    scores, parity = {}, {}
    metrics = [(row["id"], set(row["gold_ids"])) for row in fixture]
    groups = shared_evidence_clusters([{"gold": sorted(v)} for _, v in metrics])
    baseline = [target <= set(found[id_]["fp16"]) for id_, target in metrics]
    for n in LIMITS:
        summary = Counter()
        complete = []
        for id_, gold in metrics:
            r = found[id_]
            original, perturbed = r["fp16"], r[n]
            both = set(original) & set(perturbed)
            summary.update(ordered=original == perturbed, identical_set=len(both) == 64,
                           first=original[0] == perturbed[0], shared_documents=len(both))
            complete.append(gold <= set(perturbed))
        scores[n] = {"baseline_complete": sum(baseline), "quantized_complete": sum(complete),
                     "paired": compare(complete, baseline, groups)}
        parity[n] = dict(summary) | {"mean_shared_top64": summary["shared_documents"]/len(fixture)}
    amounts = np.asarray(timings)
    assert amounts.shape == (128, 2) and np.all(amounts > 0)
    change = amounts[:, 1] - amounts[:, 0]
    sample = np.random.default_rng(479031).integers(0, len(change), (4000, len(change)))
    time = {"p50_ms": dict(zip(MODES, np.median(amounts, 0).tolist(), strict=True)),
            "p95_ms": dict(zip(MODES, np.quantile(amounts, .95, axis=0).tolist(), strict=True)),
            "paired_median_ms": float(np.median(change)),
            "paired_95_ms": np.quantile(np.median(change[sample], axis=1), [.025, .975]).tolist()}
    return {"device": device, "fixture_sha256": stamp, "requests": len(by_id), "measurements": len(timings),
            "med_int8_build_ms": median(r["int8_build_ms"] for r in measurement),
            "timing": time, "parity": parity, "coverage": scores, **meta}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("fixture", "output"):
        p.add_argument("--" + opt, type=Path, required=True)
    p.add_argument("--l4", nargs=4, type=Path, required=True)
    p.add_argument("--b200", nargs=4, type=Path)
    p.add_argument("--rtx6000", nargs=4, type=Path)
    a = p.parse_args()
    fixture = gzip.decompress(a.fixture.read_bytes())
    out = {}
    for name, marker in (("l4", "NVIDIA L4"), ("b200", "NVIDIA B200"), ("rtx6000", "RTX PRO 6000")):
        if inputs := getattr(a, name):
            runs = sorted(map(unpack, inputs), key=lambda c: c["shard"])
            device = runs[0]["device"]
            assert marker in device
            out[name] = run(fixture, runs, device)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
