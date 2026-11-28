#!/usr/bin/env python3
"""Replay paired evidence-pin interventions against frozen endpoint panels."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from summarize_kam_distant_ablation import continuous_interval, margin_panel
from summarize_kam_pretrained_endpoint_selection import family_difference

PROTOCOL = "kam-evidence-pins-v1"
WIDTHS = ("4096", "8192")
ARMS = ("all-evidence", "all-background", "tail-evidence", "tail-background")
BASE = ("flash", "protected", "tail-256")
CONTROLS = {d: {a: f"{d}-{{seed}}-{a}" for a in BASE} for d in ("hotpot",)}
CONTROLS["musique"] = {a: f"transfer-{{seed}}-{v}" for a, v in zip(BASE, ("flash", "protected", "selected"), strict=True)}


def read(path):
    return json.loads(path.read_text())


def archive(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def summarize_timing(profile, named):
    timing = {}
    for width in WIDTHS:
        assert set(profile["outcomes"][width]) == set(named)
        timing[width] = {}
        for name, row in profile["outcomes"][width].items():
            wall, event = map(np.asarray, (row["wall_ms"], row["cuda_ms"]))
            assert len(wall) == len(event) == 15 and all(np.isfinite(v).all() and min(v)>0 for v in (wall,event))
            timing[width][name] = {"wall_p50_ms": float(np.median(wall)), "wall_p95_ms": float(np.quantile(wall,.95)),
                                  "cuda_p50_ms": float(np.median(event)), "peak_bytes": row["incremental_allocated_bytes"]}
    return timing


def retrieval_summary(values, cases, comparisons, *, interaction=None):
    families = np.array([r["family"] for r in cases])
    linked = np.array([r["anchored_foil"] for r in cases])
    output = {}
    for cohort, keep in (("all", np.ones_like(linked)), ("linked", linked), ("unlinked", ~linked)):
        if not keep.any():
            continue
        panel = {name: a[:,keep] for name,a in values.items()}
        quality = {name: {"signed_margin": float(v.mean()), "mirrored_accuracy": float((v > 0).mean()),
                         "mirrored_by_seed": (v > 0).mean(axis=1).tolist()} for name,v in panel.items()}
        paired = {}
        for a, b in comparisons:
            x, y = panel[a], panel[b]
            paired[f"{a}_minus_{b}"] = {"signed_margin": continuous_interval(x-y, families[keep]),
                                      "mirrored": family_difference(x>0, y>0, families[keep])}
        output[cohort] = {"questions": int(keep.sum()), "quality": quality, "paired": paired}
        if interaction is not None:
            a, b, c, d = (panel[name] for name in interaction)
            def binary(x):
                return (x>0).astype(np.int8)
            output[cohort]["interaction"] = {
                "signed_margin": continuous_interval((a-b)-(c-d), families[keep]),
                "mirrored": family_difference(binary(a)-binary(b), binary(c)-binary(d), families[keep])}
    return output


def summarize(work, plan_path, m_manifest, h_manifest, m_prior, h_prior):
    plan = read(plan_path)
    assert plan["protocol"] == PROTOCOL
    plan_hash = digest(plan_path)
    profile_path = work / "evidence-pin-profile.json"
    profile = read(profile_path)
    assert (profile["plan_sha256"], profile["domain"], profile["seed"], profile["variant"]) == (
        plan_hash, "musique", SEEDS[0], "profile")
    gpu = profile["gpu"]
    hashes = {profile_path.stem: digest(profile_path)}
    result = {}
    for domain, manifest_path, prior_path in (("musique", m_manifest, m_prior), ("hotpot", h_manifest, h_prior)):
        manifest, prior = read(manifest_path), archive(prior_path)
        fixture_hash = digest(manifest_path)
        spec = plan["datasets"][domain]
        assert spec["source_manifest_sha256"] == fixture_hash and plan["tokenizer_sha256"] == manifest["tokenizer_sha256"]
        labels = [(r["id"],r["family"],r["expected"]) for r in spec["rows"]]
        assert len(labels) == len(set(labels)) == 256
        span_sizes = Counter()
        for row in spec["rows"][::2]:
            a, b = row["evidence"]
            c, d = row["background"]
            assert 0 <= b-a == d-c <= 5 and a >= 1 and c == b
            span_sizes[b-a] += 1
        metadata_key = "transfer_manifest_sha256" if domain == "musique" else "hotpot_manifest_sha256"
        outcomes = {}
        for arm in BASE:
            outcomes[arm] = []
            for seed in SEEDS:
                record = json.loads(prior[CONTROLS[domain][arm].format(seed=seed)])
                assert (record["seed"], record["gpu"], record[metadata_key]) == (seed, gpu, fixture_hash)
                outcomes[arm].append(record)
        checks = {r["seed"]: r["checkpoint_sha256"] for r in outcomes["flash"]}
        assert profile["checkpoint_sha256"] == checks[SEEDS[0]]
        for rows in outcomes.values():
            assert all(checks[row["seed"]] == row["checkpoint_sha256"] for row in rows)
        for arm in ARMS:
            outcomes[arm] = []
            for seed in SEEDS:
                file = work / f"evidence-pins-{domain}-{seed}-{arm}.json"
                row = read(file)
                hashes[file.stem] = digest(file)
                assert (row["gpu"], row["domain"], row["seed"], row["variant"], row["plan_sha256"], row["checkpoint_sha256"]) == (
                    gpu, domain, seed, arm, plan_hash, checks[seed])
                outcomes[arm].append(row)
        comparison = (("all-evidence","all-background"), ("tail-evidence","tail-background"),
                      ("all-evidence","protected"), ("tail-evidence","protected"),
                      ("all-evidence","flash"), ("tail-evidence","flash"))
        widths = {}
        for width in WIDTHS:
            cases = manifest["fixture"][width]["questions"]
            assert [(r["id"],r["family"],i) for r in cases for i in (0,1)] == labels
            panel = {}
            for arm, rows in outcomes.items():
                records = [row["outcomes"][width] for row in rows]
                if domain == "musique" and arm in BASE:
                    records = [p["retrieval"] for p in records]
                panel[arm] = np.stack([margin_panel(r, labels, exact=True) for r in records])
            widths[width] = retrieval_summary(panel, cases, comparison, interaction=ARMS)
        result[domain] = {"manifest_sha256": fixture_hash, "prior_sha256": digest(prior_path),
                          "extra_block_count": span_sizes, "widths": widths}
    return {"plan_sha256": plan_hash, "raw_sha256": hashes, "gpu": gpu,
            "panels": result, "timing": summarize_timing(profile, (*BASE, *ARMS))}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for path in ("work", "plan", "musique_manifest", "hotpot_manifest", "musique_prior", "hotpot_prior", "output"):
        cli.add_argument(path, type=Path)
    a = cli.parse_args()
    write_json(a.output, summarize(a.work, a.plan, a.musique_manifest, a.hotpot_manifest, a.musique_prior, a.hotpot_prior))
