#!/usr/bin/env python3
"""Relate raw-text training KL to drift and correctness at evaluation cues."""
import argparse
import json
from pathlib import Path

import numpy as np


def ranks(values):
    order = np.argsort(values, kind="stable")
    values = np.asarray(values)[order]
    rank = np.empty(len(order), dtype=float)
    first = 0
    while first < len(order):
        last = first + 1
        while last < len(order) and values[last] == values[first]:
            last += 1
        rank[order[first:last]] = (first + last - 1) / 2
        first = last
    return rank


def rank_correlation(x, y):
    a, b = ranks(x), ranks(y)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(raw, factorial):
    output = {}
    assert {(m["model"], m["task"]) for m in raw} == {
        (model, task) for model in factorial for task in ("gpqa", "mmlu")
    }
    for result in raw:
        model, task = result["model"], result["task"]
        official = factorial[model]
        train = official["training"]
        assert result["revision"] == official["revision"]
        assert result["panel"]["source_sha256"] == official["tasks"][task]["source_sha256"]
        assert result["runs"].keys() == train.keys()
        teacher = np.asarray(result["teacher_choice_log_p"])
        gold = np.asarray(result["panel"]["gold"])
        base_choice = np.asarray(result["untouched_choices"])
        assert np.array_equal(teacher.argmax(-1), base_choice)
        base = base_choice == gold
        rows = {}
        for name, x in result["runs"].items():
            assert x["adapter_sha256"] == train[name]["adapter_sha256"]
            flags = np.asarray([r["correct"] for r in x["rows"]], dtype=bool)
            teacher_flags = np.asarray([r["teacher_correct"] for r in x["rows"]], dtype=bool)
            full_kl = np.asarray([r["full_kl"] for r in x["rows"]])
            log_prob = np.asarray([r["choice_log_p"] for r in x["rows"]])
            choices = np.asarray([r["choice"] for r in x["rows"]])
            assert np.array_equal(teacher_flags, base)
            assert np.array_equal(flags, choices == gold)
            assert (full_kl >= -1e-5).all() and np.isfinite(full_kl).all()
            assert np.array_equal(log_prob.argmax(-1), choices)
            choice_kl = (np.exp(teacher) * (teacher - log_prob)).sum(-1)
            rows[name] = {
                "n": len(gold), "correct": int(flags.sum()), "reference_correct": int(base.sum()),
                "lost": int((base & ~flags).sum()), "gained": int((flags & ~base).sum()),
                "choice_disagreement": int((choices != base_choice).sum()),
                "cue_full_kl": float(full_kl.mean()), "cue_choice_kl": float(choice_kl.mean()),
                "validation_text_kl": train[name]["training"][-1]["forward_kl"],
                "seed": train[name]["seed"], "recipe": train[name]["recipe"], "arm": train[name]["arm"],
            }
        cells = {}
        for recipe in ("reference", "high_lr_kl", "lower_lr", "lower_lr_kl"):
            for arm in ("kam", "dot"):
                fits = [r for r in rows.values() if r["recipe"] == recipe and r["arm"] == arm]
                assert len(fits) == 3
                cells[f"{recipe}-{arm}"] = {
                    key: float(np.mean([r[key] for r in fits]))
                    for key in ("cue_full_kl", "cue_choice_kl", "validation_text_kl",
                                "correct", "lost", "gained", "choice_disagreement")
                }
        correlations = {
            k: {v: rank_correlation([r[k] for r in rows.values()], [r[v] for r in rows.values()])
                for v in ("lost", "choice_disagreement", "correct")}
            for k in ("validation_text_kl", "cue_full_kl", "cue_choice_kl")
        }
        output.setdefault(model, {})[task] = {
            "n": len(gold), "untouched_correct": int(base.sum()),
            "seed_means": cells, "per_fit": rows,
            "descriptive_spearman_across_24_related_fits": correlations,
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", nargs=4, type=Path)
    parser.add_argument("--factorial", nargs=2, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = [json.loads(p.read_text()) for p in args.probe]
    factorial = {v["model"]: v for v in (json.loads(p.read_text()) for p in args.factorial)}
    args.output.write_text(json.dumps(summarize(inputs, factorial), indent=2, sort_keys=True) + "\n")
