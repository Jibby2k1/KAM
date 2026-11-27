#!/usr/bin/env python3
"""Summarize fixed residual scales and validation-only drift caps."""
import argparse
import json
from pathlib import Path

CAPS = (0.005, 0.01, 0.02)
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


def outcomes(predictions, base, gold):
    assert len(predictions) == len(base) == len(gold)
    correct = [p == g for p, g in zip(predictions, gold, strict=True)]
    reference = [p == g for p, g in zip(base, gold, strict=True)]
    lost = sum(b and not a for a, b in zip(correct, reference, strict=True))
    gained = sum(a and not b for a, b in zip(correct, reference, strict=True))
    assert sum(correct) - sum(reference) == gained - lost
    return {"n": len(gold), "correct": sum(correct), "untouched_correct": sum(reference),
            "lost": lost, "gained": gained,
            "exact_choice_agreement": sum(a == b for a, b in zip(predictions, base, strict=True))}


def summarize(raw):
    models = {}
    for run in raw:
        models.setdefault(run["model"], []).append(run)
    assert len(models) == 2
    rows = {}
    for model, runs in models.items():
        assert {(r["recipe"], r["arm"]) for r in runs} == {
            ("reference", "kam"), ("reference", "dot"), ("high_lr_kl", "kam"), ("high_lr_kl", "dot")
        }
        reference = runs[0]
        result = {}
        for run in runs:
            assert run["seed"] == 731221
            assert run["revision"] == reference["revision"] and run["gpu"] == reference["gpu"]
            assert run["panels"] == reference["panels"] and run["untouched"] == reference["untouched"]
            assert tuple(o["scale"] for o in run["observations"]) == SCALES
            grid = []
            for obs in run["observations"]:
                val = obs["validation"]
                accuracy = {name: outcomes(obs["choices"][name], run["untouched"][name], pane["gold"])
                            for name, pane in run["panels"].items()}
                grid.append({"scale": obs["scale"], "validation_kl": val["forward_kl"],
                             "validation_nll": val["nll"], "metrics": accuracy})
            assert all(v["lost"] == v["gained"] == 0 for v in grid[0]["metrics"].values())
            selected = {str(cap): max((index for index, candidate in enumerate(grid)
                                       if candidate["validation_kl"] <= cap), default=None)
                        for cap in CAPS}
            result[f"{run['recipe']}-{run['arm']}"] = {
                "adapter_sha256": run["adapter_sha256"], "grid": grid, "validation_cap_indices": selected,
            }
        rows[model] = {"revision": reference["revision"], "seed": reference["seed"],
                       "gpu": reference["gpu"], "panels": reference["panels"], "runs": result}
    for task in ("gpqa", "mmlu"):
        intact = [r["panels"][task] for r in rows.values()]
        assert intact[0] == intact[1]
    return {"scales": SCALES, "validation_caps": CAPS, "models": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs=8, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    data = summarize([json.loads(p.read_text()) for p in a.input])
    a.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
