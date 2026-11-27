#!/usr/bin/env python3
"""Validate and aggregate the three-arm pretrained benchmark experiment."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from summarize_kam_general_benchmarks import TASKS, details, paired

ARMS = ("kam", "dot", "pretrained")


def read_archive(path):
    if path.is_dir():
        def read(name):
            return json.loads((path / f"{name}.json").read_text())
        return read
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    return data.__getitem__


def summarize(path):
    read = read_archive(path)
    checkpoints = {}
    tasks = {}
    for task, total in TASKS.items():
        code = task in ("humaneval", "mbpp")
        suffix = "-graded" if code else ""
        parts = {arm: read(f"{task}-{arm}{suffix}") for arm in ARMS}
        rows = {arm: part["records"] for arm, part in parts.items()}
        info = {arm: part["metadata"] for arm, part in parts.items()}
        for arm in ARMS:
            assert [x["index"] for x in rows[arm]] == list(range(total))
            assert info[arm]["task"] == task and info[arm]["arm"] == arm
            stable = {k: v for k, v in info[arm].items() if k not in
                      {"task", "source_sha256", "sandbox_sha256", "cache_check", "gpu"}}
            assert checkpoints.setdefault(arm, stable) == stable
        for key in ("source_sha256", "revision", "model_fingerprints", "context_tokens"):
            assert len({i[key] for i in info.values()}) == 1, key
        if code:
            assert len({i["sandbox_sha256"] for i in info.values()}) == 1
        lengths = [r["prompt_tokens"] for r in rows["kam"]]
        assert all([r["prompt_tokens"] for r in rows[arm]] == lengths for arm in ARMS[1:])
        assert max(lengths) <= info["kam"]["context_tokens"]
        tasks[task] = {
            "source_sha256": info["kam"]["source_sha256"],
            "sandbox_sha256": info["kam"].get("sandbox_sha256"),
            "prompt_token_quantiles": np.quantile(lengths, [0, .5, .95, 1]).tolist(),
            "cache_checks": {a: info[a]["cache_check"] for a in ARMS},
            "device": {a: info[a]["gpu"] for a in ARMS},
            "arms": {a: details(rows[a]) for a in ARMS},
            "paired": {
                "kam_vs_dot": paired(rows["kam"], rows["dot"]),
                "kam_vs_pretrained": paired(rows["kam"], rows["pretrained"]),
                "dot_vs_pretrained": paired(rows["dot"], rows["pretrained"]),
            },
        }
    assert checkpoints["kam"]["new_trainable_parameters"] == checkpoints["dot"]["new_trainable_parameters"]
    assert checkpoints["pretrained"]["new_trainable_parameters"] == 0
    training = [checkpoints[a]["training"] for a in ("kam", "dot")]
    assert {k:v for k,v in training[0].items() if k != "arm"} == {
        k:v for k,v in training[1].items() if k != "arm"
    }
    return {"checkpoints": checkpoints, "tasks": tasks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.input), indent=2, sort_keys=True) + "\n")
