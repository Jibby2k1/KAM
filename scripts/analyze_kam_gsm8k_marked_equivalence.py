#!/usr/bin/env python3
"""Require the marked line while allowing equivalent mathematics after it."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from grade_kam_full_math500 import check_environment, grade_prediction, parse_gold
from kam_adapter_preservation import ARMS, SEEDS, digest, key
from kam_gsm8k_format_effects import load_fixture
from kam_math_kl_protocol import MARKERS, test_runs, write_json
from summarize_kam_adapter_preservation import difference, read_archive
from summarize_kam_gsm8k_format_effects import gap_change


def grade_and_compare(work, raw, archive, selection):
    check_environment()
    manifest, dataset = load_fixture(work)
    read = read_archive(archive)
    selected = json.loads(selection.read_text())
    recipe = selected["selected_recipe"]
    labels = [parse_gold(row["answer"]) for row in dataset]
    assert all(len(answer) == 1 for answer in labels)
    patterns = {mode: re.compile(rf"^[ \t]*{re.escape(marker)}[ \t]+([^\r\n]*)$", re.I | re.M)
                for mode, marker in MARKERS.items()}
    vectors, checkpoints = {}, {}
    for run in test_runs(recipe):
        name = key(*run)
        path = raw / f"format-{name}.json"
        payload = json.loads(path.read_text())
        prior = read(f"scored-{name}")
        assert digest(path) == prior["metadata"]["generation_sha256"]
        raw_rows, grade_rows = payload["records"], prior["records"]
        assert [row["id"] for row in raw_rows] == [row["id"] for row in grade_rows] == manifest["ids"]
        vectors[name], checkpoints[name] = {}, {}
        for mode, pattern in patterns.items():
            found_ids, parsed_ids, correct_ids, issues = [], [], [], []
            values = {"shared": [], "strict": [], "marked_equiv": []}
            for r, g, answer in zip(raw_rows, grade_rows, labels, strict=True):
                matches = pattern.findall(r[mode]["completion"])
                # Exactly one candidate line is considered, even if its expression fails.
                standalone = {"parsed": False, "correct": False}
                if matches:
                    found_ids.append(r["id"])
                    standalone = grade_prediction(answer, matches[-1], issues)
                if standalone["parsed"]:
                    parsed_ids.append(r["id"])
                if standalone["correct"]:
                    correct_ids.append(r["id"])
                for label, passed in (("shared", g[mode]["symbolic"]["correct"]),
                                      ("strict", g[mode]["strict"]["correct"]),
                                      ("marked_equiv", standalone["correct"])):
                    values[label].append(passed)
            checkpoints[name][mode] = {"standalone_ids": found_ids, "parsed_ids": parsed_ids,
                                       "correct_ids": correct_ids, "grader_issues": issues}
            vectors[name][mode] = {label: np.array(item_flags) for label, item_flags in values.items()}

    effects = {}
    for mode in MARKERS:
        baseline = vectors[key(*test_runs(recipe)[-1])][mode]
        layouts = {label: {arm: np.stack([vectors[key(seed, recipe, arm)][mode][label] for seed in SEEDS])
                           for arm in ARMS} for label in baseline}
        scores = {label: {"untouched": float(baseline[label].mean()),
                          "scores": {arm: float(v.mean()) for arm, v in arms.items()},
                          "kam_minus_dot": difference(arms["kam"], arms["dot"])}
                  for label, arms in layouts.items()}
        effects[mode] = {"scores": scores, "shared_to_standalone": gap_change(layouts["shared"], layouts["marked_equiv"]),
                         "standalone_to_strict": gap_change(layouts["marked_equiv"], layouts["strict"])}
    return {"model": selected["model"], "n": len(dataset), "seeds": SEEDS,
            "fixture_sha256": manifest["fixture_sha256"], "checkpoints": checkpoints, "effects": effects}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fixture", type=Path)
    cli.add_argument("raw", type=Path)
    cli.add_argument("archive", type=Path)
    cli.add_argument("selection", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    write_json(args.output, grade_and_compare(args.fixture, args.raw, args.archive, args.selection))
