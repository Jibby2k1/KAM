#!/usr/bin/env python3
"""Verify the frozen replay and score both-order generalization by cohort."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, digest
from kam_math_kl_protocol import write_json
from probe_kam_routing_bottlenecks import MODES, PROTOCOL
from summarize_kam_adapter_preservation import read_archive
from summarize_kam_overnight_routes import timing, verified
from summarize_kam_pretrained_endpoint_selection import family_difference


def summarize(work, fixture, study, previous):
    manifest = json.loads(fixture.read_text())
    assert manifest["previous_outcome_sha256"] == digest(previous)
    prior = read_archive(previous)
    excluded, cases = None, None
    checkpoints, flags, hashes = {}, {}, {}
    for name, (seed, _, arm), data, fingerprint in verified(work, fixture, study, "probe", PROTOCOL):
        current_excluded = set(data["metadata"]["excluded"])
        if excluded is None:
            excluded = current_excluded
            assert excluded <= {c["id"] for c in manifest["cases"] if c["cohort"] == "fresh_dev"}
            cases = [row for row in manifest["cases"] if row["id"] not in excluded]
        assert current_excluded == excluded
        records = data["records"]
        assert [(r["id"], r["expected"]) for r in records] == [(c["id"], i) for c in cases for i in (0, 1)]
        old = defaultdict(set)
        for row in prior("routes-" + name)["records"]:
            if row["dense"]["first_correct"]:
                old[row["question"], row["expected"]].add(row["dense"]["margin"])
        by_id = {c["id"]: c for c in cases}
        for row in records:
            assert set(row["scores"]) == set(MODES[arm]) and row["tokens"] > 0
            if by_id[row["id"]]["cohort"].startswith("replay_"):
                assert old[row["id"], row["expected"]] == {row["scores"]["native"]["margin"]}
            for value in row["scores"].values():
                margin = value["margin"]
                assert np.isfinite(margin) and np.isfinite(value["median_ms"]) and value["median_ms"] > 0
                assert value["correct"] == (margin != 0 and int(margin > 0) == row["expected"])
        checkpoints[name], hashes[name] = {}, fingerprint
        for cohort in ("replay_train", "replay_dev", "fresh_dev", "fresh_dev_anchor"):
            ids = [c["id"] for c in cases if c["cohort"] == cohort or
                   (cohort == "fresh_dev_anchor" and c["cohort"] == "fresh_dev" and c["foil_anchor"])]
            cohort_cases = set(ids)
            selected = [r for r in records if r["id"] in cohort_cases]
            stats = {}
            for mode in MODES[arm]:
                scores = [r["scores"][mode] for r in selected]
                signs = np.sign([s["margin"] for s in scores]).reshape(-1, 2)
                accurate = np.array([s["correct"] for s in scores]).reshape(-1, 2).all(axis=1)
                flags[seed, arm, cohort, mode] = accurate
                stats[mode] = {"both_correct": int(accurate.sum()), "same_letter_both_orders": int(
                    ((signs[:, 0] == signs[:, 1]) & (signs[:, 0] != 0)).sum()),
                    "ties": int((signs == 0).sum()), **timing([s["median_ms"] for s in scores])}
            if arm == "dot":
                margins = [(r["scores"]["native"]["margin"], r["scores"]["explicit_dense"]["margin"]) for r in selected]
                stats["native_explicit_parity"] = {
                    "same_choice": int(sum(np.sign(a) == np.sign(b) for a, b in margins)),
                    "total": len(margins), "max_absolute_margin_delta": max(abs(a-b) for a, b in margins),
                }
            checkpoints[name][cohort] = stats
    summary = {}
    for cohort in ("replay_train", "replay_dev", "fresh_dev", "fresh_dev_anchor"):
        families = [c["family"] for c in cases if c["cohort"] == cohort or
                    (cohort == "fresh_dev_anchor" and c["cohort"] == "fresh_dev" and c["foil_anchor"])]
        out = {"questions": len(families), "families": len(set(families)), "arms": {}}
        for arm in (*ARMS, "pretrained"):
            modes = MODES[arm]
            seeds = SEEDS if arm in ARMS else (0,)
            matrices = {mode: np.stack([flags[seed, arm, cohort, mode] for seed in seeds]) for mode in modes}
            ref = "explicit_dense" if arm == "dot" else "native"
            out["arms"][arm] = {m: {"accuracy": float(a.mean()), "against_native": family_difference(a, matrices["native"], families),
                                  "against_matched_compute": family_difference(a, matrices[ref], families)}
                                for m, a in matrices.items()}
        summary[cohort] = out
    return {"manifest_sha256": digest(fixture), "outcome_sha256": hashes, "excluded": sorted(excluded),
            "cohorts": summary, "checkpoints": checkpoints}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "fixture", "study", "previous", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write_json(args.output, summarize(args.work, args.fixture, args.study, args.previous))
