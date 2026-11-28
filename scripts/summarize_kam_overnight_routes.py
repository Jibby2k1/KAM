#!/usr/bin/env python3
"""Verify frozen checkpoints and replay overnight routing and retirement results."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, digest, key
from kam_math_kl_protocol import test_runs, write_json
from kam_musique_live_routes import ROUTE_SETS, SHORT_MIRROR_PROTOCOL
from kam_musique_native_routes import PROTOCOL as NATIVE_PROTOCOL, ROUTES as NATIVE_ROUTES
from probe_kam_batch_retirement import PROTOCOL as RETIREMENT_PROTOCOL
from summarize_kam_adapter_preservation import read_archive
from summarize_kam_pretrained_endpoint_selection import family_difference

BUDGETS_MS = (15, 25, 40, 60, 100, 150)
IDENTITY = ("model", "model_fingerprints", "revision", "gpu", "adapter_sha256", "training")


def verified(work, manifest_path, study, prefix, protocol):
    selection = study / "llama3-selection.json"
    recipe = json.loads(selection.read_text())["selected_recipe"]
    original = read_archive(study / "llama3-outcomes.json.gz")
    for run in test_runs(recipe):
        name = key(*run)
        path = work / f"{prefix}-{name}.json"
        data = json.loads(path.read_text())
        meta = data["metadata"]
        assert meta["protocol"] == protocol and meta["manifest_sha256"] == digest(manifest_path)
        assert meta["selection_sha256"] == digest(selection)
        assert (meta["seed"], meta["recipe"], meta["arm"]) == run
        ref = original(f"test-{name}")["metadata"]
        for field in IDENTITY:
            assert meta[field] == ref[field], (name, field)
        yield name, run, data, digest(path)


def timing(milliseconds):
    return dict(zip(("p50_ms", "p95_ms"), np.quantile(milliseconds, (.5, .95)).tolist(), strict=True))


def calibration_bias(path):
    archive = read_archive(path)
    biases = {}
    study = json.loads(path.with_name("both-earlier-manifest.json").read_text())
    recipe = "lower_lr_kl05"
    for run in test_runs(recipe):
        name = key(*run)
        data = archive(f"endpoint-{name}")
        rows = data["records"]
        assert data["metadata"]["manifest_sha256"] == digest(path.with_name("both-earlier-manifest.json"))
        assert len(rows) == 4 * len(study["cases"])
        scales = []
        for left, right in zip(rows[::2], rows[1::2], strict=True):
            assert left["question"] == right["question"] and not left["reverse"] and right["reverse"]
            for mode in ("true_first", "true_second"):
                scales.append((left[mode]["b_minus_a"] + right[mode]["b_minus_a"]) / 2)
        biases[name] = float(np.median(scales))
    return biases


def live(work, manifest_path, study, calibration=None):
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest["protocol"]
    routes = NATIVE_ROUTES if protocol == NATIVE_PROTOCOL else ROUTE_SETS[protocol]
    reference = {SHORT_MIRROR_PROTOCOL: "mirrored", NATIVE_PROTOCOL: NATIVE_ROUTES[0]}.get(protocol, "direct")
    assert tuple(manifest["routes"]) == routes
    cases, hashes, checkpoint, scores = {}, {}, {}, {},
    excluded = None
    biases = calibration_bias(calibration) if calibration else {}
    modes = (*routes, "calibrated_direct") if biases else routes
    for name, run, data, fingerprint in verified(work, manifest_path, study, "routes", protocol):
        omissions = set(data["metadata"]["excluded_pairs"])
        if excluded is None:
            excluded = omissions
            assert omissions <= {c["pair"] for c in manifest["cases"]}
            cases = {c["pair"]: c for c in manifest["cases"] if c["pair"] not in omissions}
        assert omissions == excluded
        presented = [
            (c["pair"], c["family"], q, reverse, i ^ reverse, slot)
            for c in cases.values() for i, q in enumerate(c["questions"])
            for reverse in (False, True) for slot in (0, 1)
        ]
        rows = data["records"]
        fields = ("pair", "family", "question", "reverse", "expected", "earlier_slot")
        assert [tuple(r[f] for f in fields) for r in rows] == presented
        hashes[name], checkpoint[name], scores[run] = fingerprint, {}, {}
        for split in ("dev", "train"):
            selected = [r for r in rows if cases[r["pair"]]["split"] == split]
            assert selected and len(selected) % 4 == 0
            checkpoint[name][split], scores[run][split] = {}, {}
            for route in modes:
                calibrated = route == "calibrated_direct"
                observed = [r["direct" if calibrated else route] for r in selected]
                for item, row in zip(observed, selected, strict=True):
                    assert np.isfinite(item["margin"]) and np.isfinite(item["median_ms"])
                    assert item["median_ms"] > 0 and item["tokens"] > 0
                    assert item["correct"] == (item["margin"] != 0 and int(item["margin"] > 0) == row["expected"])
                correct = np.array([
                    int(o["margin"] > biases[name]) == r["expected"] and o["margin"] != biases[name]
                    if calibrated else o["correct"] for o, r in zip(observed, selected, strict=True)
                ]).reshape(-1, 4)
                ms = np.array([o["median_ms"] for o in observed]).reshape(-1, 4)
                scores[run][split][route] = correct.all(axis=1)
                checkpoint[name][split][route] = {
                    "all_four_correct": int(correct.all(axis=1).sum()),
                    "presentation_correct": int(correct.sum()),
                    **timing(ms),
                    "mean_prompt_tokens": float(np.mean([o["tokens"] for o in observed])),
                    "all_four_correct_at_ms": {
                        str(budget): int((correct & (ms <= budget)).all(axis=1).sum()) for budget in BUDGETS_MS
                    },
                }
                if not calibrated and "first_correct" in observed[0]:
                    first = np.array([o["first_correct"] for o in observed]).reshape(-1, 4)
                    checkpoint[name][split][route]["first_correct_all_placements"] = int(first.all(axis=1).sum())
                    if protocol == NATIVE_PROTOCOL:
                        joint = correct & first
                        checkpoint[name][split][route]["joint_first_and_endpoint"] = int(joint.all(axis=1).sum())
                        checkpoint[name][split][route]["joint_at_ms"] = {
                            str(budget): int((joint & (ms <= budget)).all(axis=1).sum()) for budget in BUDGETS_MS
                        }
    recipe = json.loads((study / "llama3-selection.json").read_text())["selected_recipe"]
    contrasts = {}
    for split in ("dev", "train"):
        families = [c["family"] for c in cases.values() if c["split"] == split for _ in c["questions"]]
        contrasts[split] = {"questions": len(families), "families": len(set(families))}
        values = {route: {arm: np.stack([scores[seed, recipe, arm][split][route] for seed in SEEDS])
                          for arm in ARMS} for route in modes}
        for route, arms in values.items():
            contrasts[split][route] = {
                "kam_minus_dot": family_difference(arms["kam"], arms["dot"], families),
                "arm_accuracy": {arm: float(matrix.mean()) for arm, matrix in arms.items()},
                f"against_{reference}": {arm: family_difference(matrix, values[reference][arm], families)
                                          for arm, matrix in arms.items()},
            }
    return {"manifest_sha256": digest(manifest_path), "excluded_pairs": sorted(excluded),
            "checkpoints": checkpoint, "contrasts": contrasts, "outcome_sha256": hashes,
            **({"calibration_sha256": digest(calibration), "calibration_bias": biases} if calibration else {})}


def retirement(work, manifest_path, study):
    checkpoints, hashes = {}, {}
    prior_ids = {row["id"] for row in json.loads(manifest_path.read_text())["cases"]}
    sample_ids = None
    for name, _, data, fingerprint in verified(work, manifest_path, study, "retire", RETIREMENT_PROTOCOL):
        checkpoints[name], hashes[name] = {}, fingerprint
        assert set(data["widths"]) == {"8", "32"}
        for width, result in data["widths"].items():
            rows = result["records"]
            ids = [r["id"] for r in rows]
            sample_ids = ids if sample_ids is None else sample_ids
            assert ids == sample_ids and set(ids) <= prior_ids and len(set(ids)) == 32
            fixed, compact = (result["counters"][mode] for mode in ("fixed", "physical_retirement"))
            assert fixed["compactions"] == 0 and 0 < compact["token_slots"] <= fixed["token_slots"]
            assert all(0 < item["active_steps"] <= item["token_slots"] for item in (fixed, compact))
            elapsed = {mode: float(np.median(seconds)) for mode, seconds in result["seconds"].items()}
            assert all(len(x) == 2 and min(x) > 0 for x in result["seconds"].values())
            checkpoints[name][width] = {
                "seconds": elapsed, "exact_count": sum(r["exact"] for r in rows), "count": len(rows),
                "fixed_token_slots": fixed["token_slots"], "compacted_token_slots": compact["token_slots"],
                "compactions": compact["compactions"],
                "mismatched_ids": sorted(r["id"] for r in rows if not r["exact"]),
            }
    return {"manifest_sha256": digest(manifest_path), "checkpoints": checkpoints,
            "outcome_sha256": hashes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("live", "retirement"))
    for name in ("work", "manifest", "study", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if args.kind == "live":
        result = live(args.work, args.manifest, args.study, args.calibration)
    else:
        result = retirement(args.work, args.manifest, args.study)
    write_json(args.output, result)
