#!/usr/bin/env python3
"""Audit CPU support-ID routing when all support keys are bitwise identical."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch

from kam.memory.routers import ExactTopKRouter, pairwise_scores


@torch.inference_mode()
def audit():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    rng = torch.Generator().manual_seed(172570)
    rows = []
    for batch in (1, 4, 16):
        for count in (6, 16, 64):
            split, changed_ids, fp64_split, explicit_split, maximum = 0, 0, 0, 0, 0.0
            digest = sha256()
            router = ExactTopKRouter(3, tie_breaking="support_id")
            support_ids = torch.arange(count)
            for _ in range(32):
                q = torch.randn(batch, 8, generator=rng)
                key = torch.randn(1, 8, generator=rng)
                bank = key.repeat(count, 1)
                digest.update(q.numpy().tobytes() + key.numpy().tobytes())
                scores = pairwise_scores(q, bank)
                margin = scores.amax(-1) - scores.amin(-1)
                split += int(margin.ne(0).sum())
                maximum = max(maximum, float(margin.max()))
                chosen = router(q, bank, support_ids).indices
                changed_ids += int(chosen.ne(support_ids[:3]).any(-1).sum())
                fp64 = pairwise_scores(q.double(), bank.double())
                fp64_split += int(fp64.ne(fp64[:, :1]).any(1).sum())
                explicit = (q[:, None, :] * bank[None, :, :]).sum(-1)
                explicit_split += int(explicit.ne(explicit[:, :1]).any(1).sum())
            rows.append({"batch": batch, "supports": count, "tested": 32*batch, "scores_split": split,
                         "noncanonical_id": changed_ids, "fp64_score_split": fp64_split,
                         "explicit_reduction_split": explicit_split, "largest_score_range": maximum,
                         "input_sha256": digest.hexdigest()})
    return {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "cpu", "data": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    a.output.write_text(json.dumps(audit(), indent=2) + "\n")


if __name__ == "__main__":
    main()
