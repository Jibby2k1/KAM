#!/usr/bin/env python3
"""Evaluate frozen semantic evidence retrieval on matched serial duration swaps."""

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from audit_kam_unseentimeqa_serial import DURATION, TARGET
from run_kam_natural_feasibility import Encoder

SCENARIOS = ("original", "sham", "intervention")


def windows(events):
    return [" ".join(events[max(0, i - 2):i] + [f"Candidate event: {text}"] + events[i+1:i+3])
            for i, text in enumerate(events)]


@torch.inference_mode()
def evaluate(rows, encoder):
    q = [r["question"] for r in rows]
    target = [f"Where is {TARGET.split(text, maxsplit=1)[1].strip()}" for text in q]
    query = encoder(q + target, "query")
    scopes = [row["variants"][label]["events"] for row in rows for label in SCENARIOS]
    lengths = [len(s) for s in scopes]
    single = encoder([line for events in scopes for line in events], "passage").split(lengths)
    context = encoder([line for events in scopes for line in windows(events)], "passage").split(lengths)
    masking = [[DURATION.sub("a duration", text) for text in row["variants"]["original"]["events"]] for row in rows]
    invisible = encoder([line for events in masking for line in events], "passage").split(list(map(len, masking)))
    output = []
    for i, (row, neutral) in enumerate(zip(rows, invisible, strict=True)):
        prompt = query[[i, i + len(rows)]]
        annulled = prompt[0] @ neutral.T
        alternatives = {}
        for j, label in enumerate(SCENARIOS):
            index = i * len(SCENARIOS) + j
            one, many = prompt @ single[index].T, prompt @ context[index].T
            scores = {"direct": one[0], "target_clause": one[1], "window": many[0],
                      "window_target_clause": many[1], "duration_masked": annulled}
            gold = row["variants"][label]["gold"]
            alternatives[label] = {}
            for mode, affinity in scores.items():
                ranked = affinity.argsort(descending=True, stable=True).tolist()
                alternatives[label][mode] = {"prediction": ranked[0], "gold_rank": ranked.index(gold) + 1}
        output.append({"id": row["id"], "scenarios": alternatives})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "fixture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--shards", required=True, type=int)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    torch.set_float32_matmul_precision("highest")
    start = perf_counter()
    fixture = json.loads(args.fixture.read_text())
    rows = fixture["rows"][args.shard::args.shards]
    result = evaluate(rows, Encoder(args.assets / "encoder", 96))
    payload = {"fixture_sha256": digest(args.fixture), "models": json.loads((args.assets / "revisions.json").read_text()),
               "shard": args.shard, "shards": args.shards, "device": torch.cuda.get_device_name(),
               "elapsed_seconds": perf_counter() - start, "rows": result}
    args.output.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
