#!/usr/bin/env python3
"""Compare mirrored HotpotQA retrieval and end-to-end prefill at matched widths."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from prepare_kam_long_hotpot_transfer import PER_COHORT, PROTOCOL, WIDTHS, read
from summarize_kam_distant_ablation import margin_panel
from summarize_kam_pretrained_endpoint_selection import family_difference

ARMS = ("flash", "protected", "tail-256")


def summarize(work, fixture_path, audit_path, baseline_path):
    manifest = read(fixture_path)
    assert manifest["protocol"] == PROTOCOL
    fixture_hash = digest(fixture_path)
    with gzip.open(baseline_path, "rt") as stream:
        original = json.load(stream)
    checks = {s: json.loads(original[f"backbone-{s}-0"])["checkpoint_sha256"] for s in SEEDS}
    profile_path = work/"hotpot-profile.json"
    profile = read(profile_path)
    gpu = profile["gpu"]
    assert (profile["hotpot_manifest_sha256"], profile["seed"], profile["checkpoint_sha256"]) == (
        fixture_hash, SEEDS[0], checks[SEEDS[0]])
    hashes = {profile_path.stem: digest(profile_path)}
    audit = read(audit_path)
    hits = {x for s in audit.values() for x in s["matches"]["test"]}
    outputs, timing = {}, {}
    for width in map(str, WIDTHS):
        cases = manifest["fixture"][width]["questions"]
        labels = [(r["id"], r["family"], slot) for r in cases for slot in (0, 1)]
        families = [r["family"] for r in cases]
        assert len(families) == len(set(families)) == 2*PER_COHORT
        linked = np.array([r["anchored_foil"] for r in cases])
        clean = np.array([not hits.intersection([*r["support_ids"], r["foil_id"]]) for r in cases])
        assert linked.sum() == PER_COHORT
        values = {}
        for arm in ARMS:
            rows = []
            for seed in SEEDS:
                p = work/f"hotpot-{seed}-{arm}.json"
                d = read(p)
                hashes[p.stem] = digest(p)
                assert (d["seed"], d["arm"], d["gpu"], d["checkpoint_sha256"], d["hotpot_manifest_sha256"]) == (
                    seed, arm, gpu, checks[seed], fixture_hash)
                rows.append(margin_panel(d["outcomes"][width], labels, exact=True)>0)
            values[arm] = np.stack(rows)
        panel = {}
        for cohort, keep in (("all", np.ones_like(linked)), ("linked", linked), ("unlinked", ~linked), ("clean", clean)):
            subset = {arm: v[:,keep] for arm, v in values.items()}
            units = [f for f, include in zip(families, keep, strict=True) if include]
            panel[cohort] = {"n": len(units), "rates": {arm: {"accuracy": float(v.mean()), "by_seed": v.mean(axis=1).tolist()}
                             for arm, v in subset.items()},
                "tail_minus": {arm: family_difference(subset["tail-256"], subset[arm], units) for arm in ("flash","protected")}}
        outputs[width] = panel
        timing[width] = {}
        for arm, row in profile["outcomes"][width].items():
            wall, event = map(np.asarray, (row["wall_ms"], row["cuda_ms"]))
            assert len(wall) == len(event) == 15 and all(np.isfinite(x).all() and min(x)>0 for x in (wall,event))
            timing[width][arm] = {"wall_p50_ms": float(np.median(wall)), "wall_p95_ms": float(np.quantile(wall,.95)),
                                  "cuda_p50_ms": float(np.median(event)), "peak_bytes": row["incremental_allocated_bytes"]}
    return {"source_split": manifest["source_split"], "gpu": gpu, "manifest_sha256": fixture_hash, "raw_sha256": hashes,
            "source_audit_sha256": digest(audit_path), "quality": outputs, "timing": timing}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("work","fixture","audit","baseline","summary"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    write_json(a.summary,summarize(a.work,a.fixture,a.audit,a.baseline))
