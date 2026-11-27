#!/usr/bin/env python3
"""Extract numeric descriptors from the frozen GSM8K reference annotations."""
import argparse
import json
import re
from pathlib import Path

from run_kam_general_benchmarks import digest, load_cases

CALCULATION = re.compile(r"<<([^<>]+)>>")


def extract(data):
    source, cases = load_cases(data, "gsm8k")
    rows = []
    for index, case in enumerate(cases):
        expressions = [value.rsplit("=", 1)[0] for value in CALCULATION.findall(case["answer"])]
        rows.append({"index": index, "calculator_annotations": len(expressions),
                     "reference_division": any("/" in item for item in expressions),
                     "reference_multiplication": any("*" in item for item in expressions)})
    assert len(rows) == 1319
    return {"source_sha256": digest(source), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = extract(args.data)
    args.output.write_text(json.dumps(out, separators=(",", ":")) + "\n")
