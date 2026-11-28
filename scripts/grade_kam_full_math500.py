#!/usr/bin/env python3
"""Grade marker-free MATH-500 continuations with a shared Math-Verify protocol."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from kam_adapter_preservation import digest
from kam_math_kl_protocol import SOURCES, write_json

GRADER_VERSIONS = {
    "math-verify": "0.9.0",
    "latex2sympy2_extended": "1.11.0",
    "antlr4-python3-runtime": "4.13.2",
    "sympy": "1.14.0",
    "mpmath": "1.3.0",
}
GRADER_PROTOCOL = "math-verify-first-match-v1"
LATEX = LatexExtractionConfig(boxed_match_priority=50)
PREDICTION = (LATEX, ExprExtractionConfig())
PARSE = {"fallback_mode": "no_fallback", "extraction_mode": "first_match", "parsing_timeout": 3}
VERIFY = {"strict": True, "float_rounding": 6, "numeric_precision": 15,
          "allow_set_relation_comp": True, "timeout_seconds": 3}


def check_environment():
    actual = {name: version(name) for name in GRADER_VERSIONS}
    assert actual == GRADER_VERSIONS, actual


def parse_gold(answer):
    return parse(f"${answer}$", extraction_config=(LATEX,), **PARSE)


def grade_prediction(gold, completion):
    answer = parse(completion, extraction_config=PREDICTION, **PARSE)
    return {"parsed": bool(answer), "correct": bool(answer and verify(gold, answer, **VERIFY))}


def terminal_lines(completion):
    return "\n".join([line for line in completion.splitlines() if line.strip()][-3:])


def audit_examples():
    cases = (
        ("fraction-decimal", r"\frac{1}{2}", "0.5", True),
        ("factored-expression", r"x^2+2x+1", r"$(x+1)^2$", True),
        ("radical", r"\sqrt{8}", r"\boxed{2\sqrt{2}}", True),
        ("reordered-set", r"\{2,3\}", r"The possibilities are $\{3,2\}$.", True),
        ("interval-inequality", r"[2,3)", r"$2\leq x<3$", True),
        ("wrong-endpoints", r"[2,3)", r"$(2,3]$", False),
        ("reversed-pair", r"(2,3)", r"$(3,2)$", False),
        ("inexact-third", r"\frac{1}{3}", "0.333", False),
        ("percentage", "0.5", "50%", True),
        ("wrong-final", "13", "The denominator is 13.\nThe final answer is 12.", False),
        ("unmarked", "13", "Thus, 13.", True),
        ("no-candidate", "13", "No solution was found.", False),
    )
    outputs = {}
    for name, gold, text, expected in cases:
        output = grade_prediction(parse_gold(gold), text)
        assert output["correct"] == expected, name
        outputs[name] = output
    return outputs


def reference_data(source):
    source_sha = digest(source)
    assert source_sha == SOURCES["math500-test.jsonl"]
    data = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(data) == len({row["unique_id"] for row in data}) == 500
    return data


def audit_gold(source):
    check_environment()
    data = reference_data(source)
    kinds = Counter()
    for row in data:
        gold = parse_gold(row["answer"])
        assert len(gold) == 1 and verify(gold, gold, **VERIFY), row["unique_id"]
        kinds[type(gold[0]).__name__] += 1
    return {"source_sha256": digest(source), "self_verifying_answers": len(data),
            "symbolic_types": dict(kinds), "versions": GRADER_VERSIONS,
            "protocol": GRADER_PROTOCOL, "authored_examples": audit_examples()}


def grade_file(source, input_path, output_path):
    check_environment()
    data = reference_data(source)
    payload = json.loads(input_path.read_text())
    assert payload["metadata"]["source_sha256"] == digest(source)
    rows = payload["records"]
    assert [row["unique_id"] for row in data] == [r["id"] for r in rows]
    graded = []
    for item, record in zip(data, rows, strict=True):
        answer = parse_gold(item["answer"])
        assert len(answer) == 1
        completion = record["completion"]
        prefix = record["prefix"]
        full = grade_prediction(answer, completion)
        graded.append({
            "id": record["id"], "subject": item["subject"], "level": item["level"],
            "prompt_tokens": record["prompt_tokens"], "output_tokens": record["output_tokens"],
            "stop": record["stop"], "full": full,
            "prefix": full if prefix is None else grade_prediction(answer, prefix),
            "terminal": grade_prediction(answer, terminal_lines(completion)),
        })
    write_json(output_path, {"metadata": {**payload["metadata"], "generation_sha256": digest(input_path),
                                          "grader_protocol": GRADER_PROTOCOL, "grader_versions": GRADER_VERSIONS},
                            "records": graded})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-gold")
    audit.add_argument("source", type=Path)
    audit.add_argument("output", type=Path)
    grade = sub.add_parser("grade")
    grade.add_argument("source", type=Path)
    grade.add_argument("input", type=Path)
    grade.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.command == "audit-gold":
        write_json(args.output, audit_gold(args.source))
    else:
        grade_file(args.source, args.input, args.output)
