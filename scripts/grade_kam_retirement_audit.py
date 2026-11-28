#!/usr/bin/env python3
"""Grade changed retirement completions and strip the raw texts."""
import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "work", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    grade(args.raw, args.work, args.output)
