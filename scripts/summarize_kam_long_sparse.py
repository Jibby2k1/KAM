#!/usr/bin/env python3
"""Verify long-context checkpoint provenance and compare model quality and GPU cost."""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from summarize_kam_live_token_factorial import estimate
from summarize_kam_pretrained_endpoint_selection import family_difference

MODES = ("soft-linear", "top4-linear")


def summarize_profile(payload):
    result = {}
    for shape, modes in payload["trials"].items():
        assert set(modes) == {"flash", "sparse"}
        series = {}
        for mode, data in modes.items():
            wall, gpu = (np.asarray(data[f"{label}_ms"]) for label in ("wall", "cuda"))
            assert wall.shape == gpu.shape and len(wall) >= 15 and np.isfinite(wall).all() and np.isfinite(gpu).all()
            assert min(wall) > 0 and min(gpu) > 0 and data["incremental_allocated_bytes"] >= 0
            series[mode] = {"cuda_p50_ms": float(np.median(gpu)), "wall_p50_ms": float(np.median(wall)),
                            "wall_p95_ms": float(np.quantile(wall, .95)), "memory_bytes": data["incremental_allocated_bytes"]}
        result[shape] = series
    return result


def read_panel(result, book_ids, labels):
    books, records = result["pg19"], result["retrieval"]
    assert [r["id"] for r in books] == book_ids
    values = np.array([r["nll"] for r in books])
    assert values.shape == (len(book_ids), 3) and np.isfinite(values).all() and (values > 0).all()
    assert [(r["id"], r["family"], r["expected"]) for r in records] == labels
    margins = np.array([r["margin"] for r in records])
    flags = np.array([r["correct"] for r in records], dtype=bool)
    assert np.isfinite(margins).all() and np.array_equal(flags, (margins != 0) & ((margins > 0) == [label[2] for label in labels]))
    return values, margins, flags.reshape(-1, 2).all(axis=1)


def summarize_quality(data, pairs, overlaps):
    losses, retrieval, margins = {}, {}, {}
    for mode in (*MODES, "pretrained"):
        panels = [data[seed, mode] for seed in (SEEDS if mode in MODES else (0,))]
        losses[mode] = np.stack([panel[0].mean(axis=1) for panel in panels])
        retrieval[mode] = np.stack([panel[2] for panel in panels])
        margins[mode] = np.stack([panel[1].reshape(-1, 2) for panel in panels])
    subsets = {"overall": np.ones(len(pairs), dtype=bool),
               **{split: np.array([r["split"] == split for r in pairs]) for split in ("dev", "train")},
               "anchored_foil": np.array([r["anchored_foil"] for r in pairs]),
               "no_detected_passage_overlap": ~overlaps}
    accuracy = {}
    for label, mask in subsets.items():
        groups = {mode: a[:, mask] for mode, a in retrieval.items()}
        families = [r["family"] for r, keep in zip(pairs, mask, strict=True) if keep]
        accuracy[label] = {"n": len(families), "families": len(set(families)),
                           "arms": {mode: {"mean": float(a.mean()), "by_seed": a.mean(axis=1).tolist()}
                                    for mode, a in groups.items()},
                           "top4_minus_soft": family_difference(groups["top4-linear"], groups["soft-linear"], families)}
    mirrored, choice_diagnostics = {}, {}
    for mode, margin in margins.items():
        a, b = margin[:, :, 0], margin[:, :, 1]
        # The original assigns gold to A, and the swapped presentation assigns gold to B.
        mirrored[mode] = b > a
        choice_diagnostics[mode] = {
            "a_choice_rate": float((margin < 0).mean()),
            "tie_rate": float((margin == 0).mean()),
            "same_label_both_orders": float(((np.sign(a) == np.sign(b)) & (a != 0)).mean()),
            "single_presentation_accuracy": float(((a < 0).mean() + (b > 0).mean()) / 2),
            "mirrored_margin_accuracy": float(mirrored[mode].mean()),
        }
    return {"pg19": {"arms": {mode: {"nll": float(a.mean()), "ppl": math.exp(a.mean()), "nll_by_seed": a.mean(axis=1).tolist()}
                               for mode, a in losses.items()},
                      "top4_minus_soft_nll": estimate(losses["top4-linear"] - losses["soft-linear"])},
            "retrieval": accuracy,
            "retrieval_position_diagnostics": {"arms": choice_diagnostics,
                "mirrored_top4_minus_soft": family_difference(mirrored["top4-linear"], mirrored["soft-linear"],
                                                               [r["family"] for r in pairs])}}


def summarize(work, manifest_path, training, source_audit_path):
    with gzip.open(training, "rt") as file:
        source = json.load(file)
    manifest = json.loads(manifest_path.read_text())
    source_audit = json.loads(source_audit_path.read_text())
    previous = {name: ckpt["checkpoints"]["256"]["adapter_sha256"]
                for name, ckpt in source.items() if name.startswith("eval-") and "256" in ckpt["checkpoints"]}
    pairs = manifest["retrieval"]
    overlap_ids = {doc for source in source_audit.values() for doc in source["matches"]["test"]}
    overlaps = np.array([bool(overlap_ids.intersection([*r["support_ids"], r["foil_id"], *r["filler_ids"]])) for r in pairs])
    labels = [(r["id"], r["family"], slot) for r in pairs for slot in (0, 1)]
    book_ids = [r["id"] for r in manifest["pg19"]]
    assert len(book_ids) == len(set(book_ids)) == 48 and len(pairs) == len({r["id"] for r in pairs}) == 320
    manifest_hash = digest(manifest_path)
    kernel_file, full_file = (work / f"{n}.json" for n in ("kernel-results", "full-profile"))
    kernel, full = [json.loads(p.read_text()) for p in (kernel_file, full_file)]
    assert full["manifest_sha256"] == manifest_hash and kernel["gpu"] == full["gpu"]
    assert full["checkpoint_sha256"] == previous[f"eval-{SEEDS[0]}-top4-linear"]
    hashes = {}

    def load(kind, seed, mode):
        path = work / f"{kind}-{seed}-{mode}.json"
        result = json.loads(path.read_text())
        hashes[path.stem] = digest(path)
        assert (result["seed"], result["arm"], result["gpu"], result["manifest_sha256"]) == (seed, mode, kernel["gpu"], manifest_hash)
        assert result["checkpoint_sha256"] == previous.get(f"eval-{seed}-{mode}")
        return result

    optimized, reference, audit = {}, {}, {}
    for seed, mode in [(seed, mode) for seed in SEEDS for mode in MODES] + [(0, "pretrained")]:
        result = load("eval", seed, mode)
        optimized[seed, mode] = read_panel(result, book_ids, labels)
        backend = result["audit"]
        scored = result["scored_backend"]
        if mode == "pretrained":
            assert backend is None and scored == "flash"
            reference[seed, mode] = optimized[seed, mode]
            continue
        target = "flash" if mode == "soft-linear" else "sparse"
        explicit_scores, target_scores = [read_panel(backend["scores"][key], book_ids[:4], labels[:48])
                                          for key in ("explicit", target)]
        nll_delta = target_scores[0] - explicit_scores[0]
        choice_differences = int(np.count_nonzero(np.sign(target_scores[1]) != np.sign(explicit_scores[1])))
        max_delta = float(np.max(np.abs(nll_delta)))
        assert (backend["choice_differences"], backend["nll_max_abs_delta"]) == (choice_differences, max_delta)
        assert backend["passed"] == (choice_differences <= 1 and max_delta <= .003)
        assert scored == (target if backend["passed"] else "explicit")
        audit[f"eval-{seed}-{mode}"] = {k: v for k, v in backend.items() if k != "scores"} | {
            "scored_backend": scored, "backend_minus_explicit_mean_nll": float(nll_delta.mean())}
        ref = result if scored == "explicit" else load("reference", seed, mode)
        assert ref["scored_backend"] == "explicit"
        reference[seed, mode] = read_panel(ref, book_ids, labels)

    return {"manifest_sha256": manifest_hash, "source_audit_sha256": digest(source_audit_path),
            "retrieval_passage_overlap_ids": [r["id"] for r, overlap in zip(pairs, overlaps, strict=True) if overlap],
            "raw_sha256": {**hashes, kernel_file.stem: digest(kernel_file),
                                                                    full_file.stem: digest(full_file)},
            "gpu": kernel["gpu"], "torch": kernel["torch"], "triton": kernel["triton"], "audit": audit,
            "resident_parameter_bytes": {name: full[f"{name}_parameters_bytes"] for name in ("model", "adapter")},
            **summarize_quality(optimized, pairs, overlaps),
            "common_reference": summarize_quality(reference, pairs, overlaps),
            "kernel": summarize_profile(kernel), "full_prefill": summarize_profile(full),
            "kernel_fidelity": kernel["fidelity"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("work", "manifest", "training", "source_audit", "output"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    write_json(a.output, summarize(a.work, a.manifest, a.training, a.source_audit))
