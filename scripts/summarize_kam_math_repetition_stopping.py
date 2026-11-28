#!/usr/bin/env python3
"""Verify paired serving profiles and summarize the measured decoder work."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from kam_adapter_preservation import digest, key
from kam_math_kl_protocol import write_json
from probe_kam_math_repetition_stopping import PROTOCOL, SHAPES
from summarize_kam_adapter_preservation import read_archive


def summarize(work, manifest_path, prior_dir):
    manifest = json.loads(manifest_path.read_text())
    pair_order = sorted(manifest["pairs"], key=lambda pair: hashlib.sha256(
        f"{PROTOCOL}:{pair['integer_id']}".encode()
    ).digest())
    selection = prior_dir / "llama3-selection.json"
    prior = read_archive(prior_dir / "llama3-outcomes.json.gz")
    profiles = []
    prompt_lengths = {}
    for arm, batch, n_pairs in SHAPES:
        name = key(731221, "lower_lr_kl05", arm)
        path = work / f"timing-{batch}-{name}.json"
        data = json.loads(path.read_text())
        meta, rows = data["metadata"], data["records"]
        reference = prior(f"test-{name}")["metadata"]
        assert (meta["protocol"], meta["batch"], meta["manifest_sha256"]) == (
            PROTOCOL, batch, digest(manifest_path),
        )
        assert meta["selection_sha256"] == reference["selection_sha256"] == digest(selection)
        fields = ("arm", "seed", "recipe", "model", "model_fingerprints", "revision",
                  "gpu", "adapter_sha256", "training", "new_trainable_parameters")
        for field in fields:
            assert meta[field] == reference[field], field
        assert [row["id"] for row in rows] == [
            pair[field] for pair in pair_order[:n_pairs] for field in ("integer_id", "noninteger_id")
        ]
        widths = [row["prompt_tokens"] for row in rows]
        assert widths == prompt_lengths.setdefault(batch, widths)
        for row in rows:
            reduced = row["repeat_tokens"]
            full = row["original_tokens"]
            assert 0 < reduced <= full <= 1024
            if row["repeat_stop"] == "repetition":
                assert reduced == 384
            else:
                assert (reduced, row["repeat_stop"]) == (full, row["original_stop"])

        order = sorted(range(len(rows)), key=lambda i: (widths[i], i))
        times = meta["timings_seconds"]
        assert set(times) == {"original", "repeat"}
        assert all(len(values) == 2 and min(values) > 0 for values in times.values())
        summary = {"arm": arm, "batch": batch, "n": len(rows), "sha256": digest(path),
                   "stopped_for_repetition": sum(row["repeat_stop"] == "repetition" for row in rows)}
        for mode in ("original", "repeat"):
            counts = [rows[i][f"{mode}_tokens"] for i in order]
            summary[mode] = {
                "seconds": times[mode],
                "median_seconds": median(times[mode]),
                "output_tokens": sum(counts),
                "batch_token_slots": sum(
                    batch * max(counts[i:i+batch]) for i in range(0, len(counts), batch)
                ),
            }
        summary["elapsed_reduction_fraction"] = (
            1 - summary["repeat"]["median_seconds"] / summary["original"]["median_seconds"]
        )
        profiles.append(summary)
    return {"protocol": PROTOCOL, "manifest_sha256": digest(manifest_path), "profiles": profiles}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("manifest", type=Path)
    cli.add_argument("prior_dir", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    write_json(args.output, summarize(args.work, args.manifest, args.prior_dir))
