#!/usr/bin/env python3
"""Grade changed retirement completions and strip the raw texts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from grade_kam_full_math500 import GRADER_VERSIONS, check_environment, grade_prediction, parse_gold
from kam_adapter_preservation import digest
from kam_math_kl_protocol import write_json
from kam_tabmwp_archive_noise import load_panel
from probe_kam_batch_retirement import AUDIT_PROTOCOL


def grade(raw_path, work, output):
    check_environment()
    panel, fixture_sha = load_panel(work)
    gold = {row["id"]: parse_gold(row["answer"]) for row in panel}
    assert all(len(x) == 1 for x in gold.values())
    raw = json.loads(raw_path.read_text())
    metadata = raw["metadata"]
    assert (metadata["protocol"], metadata["manifest_sha256"]) == (AUDIT_PROTOCOL, fixture_sha)
    for result in raw["widths"].values():
        for row in result["records"]:
            if not row["exact"]:
                for mode in ("fixed", "retired"):
                    issues = []
                    answer = grade_prediction(gold[row["id"]], row.pop(f"{mode}_completion"), issues)
                    row[mode] = {**answer, "issues": issues}
    metadata.update(grading=GRADER_VERSIONS, capture_sha256=digest(raw_path))
    write_json(output, raw)


def summarize(graded, baseline, published_summary):
    registered = json.loads(published_summary.read_text())
    result, fingerprints = {}, {}
    for name, frozen_sha in registered["outcome_sha256"].items():
        prior, current = (folder / f"retire-{name}.json" for folder in (baseline, graded))
        assert frozen_sha == digest(prior)
        reference, observed = [json.loads(path.read_text()) for path in (prior, current)]
        before, after = reference["metadata"], observed["metadata"]
        assert all(after[key] == value for key, value in before.items() if key != "protocol")
        assert after["baseline_sha256"] == frozen_sha and after["protocol"] == AUDIT_PROTOCOL
        assert after["grading"] == GRADER_VERSIONS
        result[name], fingerprints[name] = {}, digest(current)
        for width, old in reference["widths"].items():
            table = Counter({"gained": 0, "lost": 0, "both_correct": 0, "both_incorrect": 0})
            rows = observed["widths"][width]["records"]
            for a, b in zip(old["records"], rows, strict=True):
                assert a == {k: v for k, v in b.items() if k not in ("fixed", "retired")}
                if not b["exact"]:
                    answer = b["fixed"]["correct"], b["retired"]["correct"]
                    labels = {(False, True): "gained", (True, False): "lost", (True, True): "both_correct",
                              (False, False): "both_incorrect"}
                    table[labels[answer]] += 1
            result[name][width] = {"outcomes": len(rows), "changed": sum(table.values()), **table}
    return {"prior_summary_sha256": digest(published_summary), "checkpoints": result, "outcome_sha256": fingerprints}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for operation, arguments in (("grade", ("raw", "work", "output")),
                                 ("summarize", ("graded", "baseline", "prior_summary", "output"))):
        command = commands.add_parser(operation)
        for name in arguments:
            command.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "grade":
        grade(args.raw, args.work, args.output)
    else:
        write_json(args.output, summarize(args.graded, args.baseline, args.prior_summary))
