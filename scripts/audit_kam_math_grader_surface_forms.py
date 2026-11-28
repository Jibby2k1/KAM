#!/usr/bin/env python3
"""Stress the frozen equivalence grader on mathematically matched surface forms."""
import argparse
from pathlib import Path

from grade_kam_full_math500 import (
    GRADER_PROTOCOL,
    GRADER_VERSIONS,
    VERIFY,
    check_environment,
    grade_prediction,
    parse_gold,
    terminal_lines,
    verify,
)
from kam_math_kl_protocol import write_json


# Each case is a gold expression, an equivalent rendering, and a plausible wrong answer.
CASES = (
    ("signed_integer", "-37", "-37", "37"),
    ("percentage", "0.625", "62.5\\%", "6.25\\%"),
    ("fraction", r"\frac{7}{16}", "0.4375", "0.437"),
    ("simplified_radical", r"\sqrt{72}", r"6\sqrt{2}", r"8\sqrt{2}"),
    ("complex_number", "-125+790i", "790i-125", "125+790i"),
    ("polynomial", "x^2-6x+9", "(x-3)^2", "(x+3)^2"),
    ("rational_function", r"\frac{19x+3}{x-7}", r"\frac{3+19x}{x-7}", r"\frac{3+19x}{x+7}"),
    ("inverse_function", r"\frac{9x+11}{x-3}", r"f^{-1}(x)=\frac{11+9x}{x-3}",
     r"f^{-1}(x)=\frac{11+9x}{x+3}"),
    ("interval", "[-4,11)", "[-4,11)", "(-4,11]"),
    ("unordered_set", r"\{-7,0,13\}", r"\{13,-7,0\}", r"\{-7,0,-13\}"),
    ("ordered_pair", "(14,-29)", "(14,-29)", "(-29,14)"),
    ("pi_multiple", r"\frac{15\pi}{4}", r"\frac{30\pi}{8}", r"\frac{15\pi}{8}"),
    ("inequality", r"t\geq -11", r"t\geq -11", "t>-11"),
)

FORMS = {
    "bare": lambda answer, _: f"The result is {answer}.",
    "delimited_math": lambda answer, _: f"The result is ${answer}$.",
    "boxed_math": lambda answer, _: "The result is $\\boxed{" + answer + "}$.",
    "unwrapped_answer_cue": lambda answer, _: f"Final answer: The final answer is {answer}. I hope this is correct.",
    "bare_after_delimited_decoy": lambda answer, decoy: f"We first considered ${decoy}$.\nSo the result is {answer}.",
    "delimited_after_delimited_decoy": lambda answer, decoy: (
        f"We first considered ${decoy}$.\nSo the result is ${answer}$."
    ),
}


def audit():
    check_environment()
    outcomes = {}
    for name, text, equivalent, wrong in CASES:
        gold = parse_gold(text)
        assert verify(gold, parse_gold(equivalent), **VERIFY), name
        assert not verify(gold, parse_gold(wrong), **VERIFY), name
        outcomes[name] = {}
        for form, render in FORMS.items():
            labels = {}
            for label, final, decoy in (("equivalent", equivalent, wrong), ("inequivalent", wrong, equivalent)):
                completion = render(final, decoy)
                labels[label] = {"full": grade_prediction(gold, completion),
                                 "terminal": grade_prediction(gold, terminal_lines(completion))}
            outcomes[name][form] = labels
    summary = {
        form: {
            label: {mode: sum(measures[form][label][mode]["correct"] for measures in outcomes.values())
                    for mode in ("full", "terminal")}
            for label in ("equivalent", "inequivalent")
        }
        for form in FORMS
    }
    reversed_inequality = verify(parse_gold(r"t\geq -11"), parse_gold(r"-11\leq t"), **VERIFY)
    return {"grader_protocol": GRADER_PROTOCOL, "versions": GRADER_VERSIONS,
            "cases_per_form_and_label": len(CASES), "accepted_counts": summary, "outcomes": outcomes,
            "reversed_inequality_verifies": bool(reversed_inequality)}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    write_json(args.output, audit())
