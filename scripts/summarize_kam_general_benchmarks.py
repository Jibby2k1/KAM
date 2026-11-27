#!/usr/bin/env python3
"""Check matched coverage and summarize per-item benchmark outcomes."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

TASKS = {"humaneval": 164, "mbpp": 500, "gpqa": 198, "mmlu": 14042, "gsm8k": 1319}
ARMS = ("T-KAM-F", "T-WIDE")


def details(rows):
    result = {"n": len(rows), "correct": sum(row["correct"] for row in rows)}
    result["score"] = result["correct"] / len(rows) if rows else None
    if rows and "outcome" in rows[0]:
        result["unit_test_outcomes"] = dict(Counter(row["outcome"] for row in rows))
    if rows and "parsed" in rows[0]:
        result["parsed_numeric"] = sum(row["parsed"] for row in rows)
    if rows and "choice" in rows[0]:
        result["answer_letter_counts"] = dict(Counter("ABCD"[row["choice"]] for row in rows))
    return result


def paired(left, right):
    delta = np.array([int(a["correct"]) - int(b["correct"])
                      for a, b in zip(left, right, strict=True)])
    gains = int(np.sum(delta == 1))
    losses = int(np.sum(delta == -1))
    n = len(delta)
    # Exact multinomial distribution of pairs under the nonparametric bootstrap.
    sims = np.random.default_rng(71001).multinomial(
        n, [gains / n, losses / n, (n - gains - losses) / n], size=20000
    )
    interval = np.quantile((sims[:, 0] - sims[:, 1]) / n, [0.025, 0.975])
    return {"delta": (gains - losses) / n, "kam_only": gains, "dot_only": losses,
            "item_bootstrap95_descriptive": interval.tolist()}


def summarize(root):
    if root.is_dir():
        def read_output(name):
            return json.loads((root / f"{name}.json").read_text())
    else:
        with gzip.open(root, "rt") as stream:
            archive = json.load(stream)

        def read_output(name):
            return archive[name]

    result = {}
    checkpoint_metadata = {}
    for task, n in TASKS.items():
        suffix = "-graded" if task in ("humaneval", "mbpp") else ""
        outputs = [read_output(f"{task}-{arm}{suffix}") for arm in ARMS]
        metadata = [output["metadata"] for output in outputs]
        left, right = [output["records"] for output in outputs]
        for arm, info, rows in zip(ARMS, metadata, [left, right], strict=True):
            assert info["task"] == task and info["arm"] == arm
            assert [row["index"] for row in rows] == list(range(n))
            shared = {key: value for key, value in info.items()
                      if key not in ("task", "source_sha256", "gpu", "sandbox_sha256")}
            assert checkpoint_metadata.setdefault(arm, shared) == shared
        assert metadata[0].get("sandbox_sha256") == metadata[1].get("sandbox_sha256")
        for key in ("source_sha256", "pair", "tokens", "seed", "data_seed", "context_bytes"):
            assert metadata[0][key] == metadata[1][key], key
        assert [row["prompt_bytes"] for row in left] == [row["prompt_bytes"] for row in right]
        width = metadata[0]["context_bytes"]
        retained = [sum(row["prompt_bytes"] <= width for row in rows) for rows in [left, right]]
        assert retained[0] == retained[1]
        result[task] = {
            "source_sha256": metadata[0]["source_sha256"],
            "sandbox_sha256": metadata[0].get("sandbox_sha256"),
            "gpu": dict(zip(ARMS, [info["gpu"] for info in metadata], strict=True)),
            "full_prompt_count": retained[0],
            "prompt_byte_quantiles": np.quantile([r["prompt_bytes"] for r in left], [0, .5, .95, 1]).tolist(),
            "arms": dict(zip(ARMS, map(details, [left, right]), strict=True)),
            "full_prompt_arms": dict(zip(
                ARMS,
                (details([row for row in rows if row["prompt_bytes"] <= width]) for rows in [left, right]),
                strict=True,
            )),
            "paired": paired(left, right),
        }
        if task == "mmlu":
            subjects = sorted({row["subject"] for row in left})
            assert len(subjects) == 57
            result[task]["by_subject"] = {
                subject: dict(zip(ARMS, (details([row for row in rows if row["subject"] == subject])
                                        for rows in [left, right]), strict=True))
                for subject in subjects
            }
    return {"checkpoints": checkpoint_metadata, "tasks": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.inputs), indent=2, sort_keys=True) + "\n")
