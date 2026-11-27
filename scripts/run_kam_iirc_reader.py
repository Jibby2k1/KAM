#!/usr/bin/env python3
"""Evaluate frozen answer decoding with fixed article identities on IIRC."""

import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re

import torch

from audit_kam_natural_data import digest
from run_kam_hotpot_pool_scaling import Reader
from run_kam_natural_feasibility import normalized_answer

ARM_SECTIONS = {"question": (None, None), "source_lead": ("lead", None),
                "article_leads": ("lead", "lead"), "linked_oracle_windows": ("lead", "oracle_window"),
                "annotated_spans": ("gold_spans", "gold_spans")}


FORMAT_PROMPTS = {"span": "Use a short phrase from the evidence.",
                  "binary": "Answer with only yes or no.",
                  "value": "Answer with only one number."}
FORMAT_POLICIES = ("generic", "universal", "question_type", "oracle_type")
NUMERIC = re.compile(r"([+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)(?:\s+(.+?))?\.?")


def question_type(question):
    text = question.lower().lstrip()
    if re.match(r"^(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", text):
        return "binary"
    if re.match(r"^how (?:many|much|long|old|far|often|high|tall)\b|^(?:what|which) (?:year|age|number|percentage|percent)\b", text):
        return "value"
    return "span"


def exact(pred, case):
    if case["answer_type"] != "value":
        return normalized_answer(pred) == normalized_answer(case["answer"])
    match = NUMERIC.fullmatch(pred.strip())
    if not match:
        return False
    number, unit = match.groups()
    if unit and normalized_answer(unit) != normalized_answer(case["answer_unit"]):
        return False
    return Decimal(number.replace(",", "")) == Decimal(case["answer"].strip())


def predict(reader, cases, format_profile=False):
    def section(doc, source, limit):
        text = f'{doc["title"]}. {doc[source]}'
        ids = reader.tokenizer(text, add_special_tokens=False, truncation=True, max_length=limit).input_ids
        return reader.tokenizer.decode(ids, skip_special_tokens=True)

    queries, rows = {}, []
    for item in cases:
        result = {}
        row = {"id": item["id"], "source_id": item["source_id"], "answer_type": item["answer_type"], "labels": result}
        rows.append(row)
        requested = ("article_leads", "annotated_spans") if format_profile else tuple(ARM_SECTIONS)
        if format_profile:
            row["inferred_type"] = question_type(item["question"])
        for arm in requested:
            src, link = ARM_SECTIONS[arm]
            parts = []
            if src:
                parts.append(section(item["source"], src, 120))
            if link:
                parts.extend(section(c, link, 165) for c in item["links"])
            generic = "Answer the question with a short answer."
            instructions = {"generic": generic}
            if format_profile:
                instructions.update({"universal": "Answer: phrase, yes/no, or digits.",
                                     "question_type": FORMAT_PROMPTS[row["inferred_type"]],
                                     "oracle_type": FORMAT_PROMPTS[item["answer_type"]]})
            prompts = {}
            trimmed = False
            while True:
                for policy, instruction in instructions.items():
                    prompt = f'{instruction} Question: {item["question"]}'
                    if parts:
                        prompt += " Evidence: " + " ".join(parts)
                    prompts[policy] = (prompt, len(reader.tokenizer(prompt).input_ids))
                excess = max(n for _, n in prompts.values()) - 512
                if excess <= 0:
                    break
                assert format_profile and len(parts) == 3
                ids = reader.tokenizer(parts[-1], add_special_tokens=False).input_ids
                assert len(ids) > excess
                parts[-1] = reader.tokenizer.decode(ids[:-(excess + 1)], skip_special_tokens=True)
                trimmed = True
            if format_profile:
                row.setdefault("matched_budget_trim", {})[arm] = trimmed
            for policy, (prompt, tokens) in prompts.items():
                key = f"{arm}:{policy}" if format_profile else arm
                result[key] = {"prompt_tokens": tokens}
                queries.setdefault(prompt, []).append((item, result[key]))
    for prediction, repeats in zip(reader.answers(list(queries)), queries.values(), strict=True):
        for item, arm in repeats:
            arm["exact"] = exact(prediction, item)
            arm["output_sha256"] = sha256(prediction.encode()).hexdigest()
            if format_profile:
                form = " ".join(normalized_answer(prediction))
                arm["output_type"] = ("binary" if form in ("yes", "no") else
                                      "value" if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)*", prediction.strip()) else "other")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--format-profile", action="store_true")
    a = p.parse_args()
    f = json.loads(a.fixture.read_text())
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    response = {"fixture_sha256": digest(a.fixture), "span_counts_sha256": f["span_counts_sha256"],
                "source_sha256": f["source_sha256"], "excluded_multispan": f["excluded_multispan"],
                "shard": a.shard, "shards": a.shards, "gpu": torch.cuda.get_device_name(),
                "assets": json.loads((a.assets / "revisions.json").read_text()),
                "rows": predict(Reader(a.assets / "reader"), f["rows"][a.shard::a.shards], a.format_profile)}
    if a.format_profile:
        response["format_policies"] = FORMAT_POLICIES
    a.output.write_text(json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
