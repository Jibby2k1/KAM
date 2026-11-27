#!/usr/bin/env python3
"""Measure support coverage and count sensitivity to displayed-date collisions."""

import argparse
import gzip
import json
from pathlib import Path

from audit_kam_natural_data import digest, words
from prepare_kam_timelineqa import DATE
from prepare_kam_timelineqa_counts import DEPTH, DONORS, SCHEMA

MODES = ("dense", "shown_month", "source_month")


def render(episode, day):
    return f"Journal date {day}. " + DATE.sub(day, episode["text"])


def rank(scores, selected):
    return [int(i) for i in scores.argsort(descending=True, stable=True).tolist() if selected[i]]


def run(a, fixture):
    import torch

    from run_kam_natural_feasibility import Encoder

    torch.set_float32_matmul_precision("highest")
    encoder = Encoder(a.assets / "encoder", 160)
    cases = fixture["personas"][a.shard::a.shards]
    results = []
    with torch.inference_mode():
        for persona in cases:
            archive = persona["episodes"]
            panel, plans, texts = [], [], []
            pairs = persona["cases"]
            for q in pairs:
                pool = [archive[i] for i in q["candidates"]]
                assert len(pool) == len(q["candidates"]) == len(set(q["candidates"])) <= DEPTH
                assert set(q["gold"]) <= set(q["candidates"])
                assert set(q["changed_dates"]).isdisjoint(q["gold"]) and len(q["changed_dates"]) == DONORS
                true_days = [e["day"] for e in pool]
                displayed = [q["changed_dates"].get(e["id"], e["day"]) for e in pool]
                for dates in (true_days, displayed):
                    local = [render(e, day) for e, day in zip(pool, dates, strict=True)]
                    plans.append((pool, time_scope(true_days, q["month"]), time_scope(dates, q["month"]), len(texts)))
                    texts.extend(local)
            embeddings = encoder(texts, "passage", max_length=96)
            vector = encoder([q["question"] for q in pairs], "query", max_length=96)
            for i, q in enumerate(pairs):
                row = {"id": q["id"], "arms": {}}
                for scenario, (pool, typed, shown, offset) in zip(("sham", "collision"), plans[2*i:2*i+2], strict=True):
                    source = [e["id"] for e in pool]
                    similarity = embeddings[offset:offset+len(pool)] @ vector[i]
                    orders = [rank(similarity, [True] * len(pool)), rank(similarity, shown), rank(similarity, typed)]
                    target = " " + " ".join(words(q["value"])) + " "
                    literal = [target in " " + " ".join(words(e["text"])) + " " for e in pool]
                    required = {e["id"] for e in pool if e["value"] == q["value"] and e["day"][:7] == q["month"]}
                    assert required == set(q["gold"])
                    row["arms"][scenario] = {
                        "ranks": {m: [source[j] for j in idx] for m, idx in zip(MODES, orders, strict=True)},
                        "full_literal_shown": [source[j] for j, match in enumerate(literal) if match and shown[j]],
                        "full_literal_source": [source[j] for j, match in enumerate(literal) if match and typed[j]],
                        "source_month": [source[j] for j, keep in enumerate(typed) if keep],
                        "shown_month": [source[j] for j, keep in enumerate(shown) if keep],
                    }
                panel.append(row)
            results.append({"seed": persona["seed"], "rows": panel})
    output = {"fixture_sha256": digest(a.fixture), "gpu": torch.cuda.get_device_name(),
              "models": json.loads((a.assets / "revisions.json").read_text()),
              "shard": a.shard, "shards": a.shards, "personas": results}
    a.output.write_bytes(gzip.compress(json.dumps(output, separators=(",", ":")).encode(), mtime=0))


def time_scope(dates, month):
    return [d[:7] == month for d in dates]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("fixture", "assets", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    fixture = json.loads(gzip.decompress(a.fixture.read_bytes()))
    assert fixture["schema"] == SCHEMA and 0 <= a.shard < a.shards
    run(a, fixture)


if __name__ == "__main__":
    main()
