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


def exact(pred, case):
    if case["answer_type"] != "value":
        return normalized_answer(pred) == normalized_answer(case["answer"])
    match = re.fullmatch(r"([+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)(?:\s+(.+?))?\.?", pred.strip())
    if not match:
        return False
    number, unit = match.groups()
    if unit and normalized_answer(unit) != normalized_answer(case["answer_unit"]):
        return False
    return Decimal(number.replace(",", "")) == Decimal(case["answer"].strip())


def predict(reader, cases):
    def section(doc, source, limit):
        text = f'{doc["title"]}. {doc[source]}'
        ids = reader.tokenizer(text, add_special_tokens=False, truncation=True, max_length=limit).input_ids
        return reader.tokenizer.decode(ids, skip_special_tokens=True)

    queries, rows = {}, []
    for item in cases:
        result = {}
        rows.append({"id": item["id"], "source_id": item["source_id"], "answer_type": item["answer_type"], "labels": result})
        for arm, (src, link) in ARM_SECTIONS.items():
            parts = []
            if src:
                parts.append(section(item["source"], src, 120))
            if link:
                parts.extend(section(c, link, 165) for c in item["links"])
            prompt = f'Answer the question with a short answer. Question: {item["question"]}'
            if parts:
                prompt += " Evidence: " + " ".join(parts)
            tokens = len(reader.tokenizer(prompt).input_ids)
            assert tokens <= 512
            result[arm] = {"prompt_tokens": tokens}
            queries.setdefault(prompt, []).append((item, result[arm]))
    for prediction, repeats in zip(reader.answers(list(queries)), queries.values(), strict=True):
        for item, arm in repeats:
            arm["exact"] = exact(prediction, item)
            arm["output_sha256"] = sha256(prediction.encode()).hexdigest()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    f = json.loads(a.fixture.read_text())
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    response = {"fixture_sha256": digest(a.fixture), "span_counts_sha256": f["span_counts_sha256"],
                "source_sha256": f["source_sha256"], "excluded_multispan": f["excluded_multispan"],
                "shard": a.shard, "shards": a.shards, "gpu": torch.cuda.get_device_name(),
                "assets": json.loads((a.assets / "revisions.json").read_text()),
                "rows": predict(Reader(a.assets / "reader"), f["rows"][a.shard::a.shards])}
    a.output.write_text(json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
