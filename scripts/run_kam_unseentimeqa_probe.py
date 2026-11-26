#!/usr/bin/env python3
"""Compare semantic, temporal, and provenance gates on exact event retrieval."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest, stable
from audit_kam_unseentimeqa import DAY, parse_events, question_key
from run_kam_hotpot_independent import read, save
from run_kam_natural_feasibility import Encoder


@torch.inference_mode()
def evaluate(rows, encoder):
    queries = encoder([r[k] for r in rows for k in ("question", "resolved_question")], "query")
    texts = [r["events"] + [r["sham"], r["collision"]] for r in rows]
    keys = encoder([text for case in texts for text in case], "passage").split(list(map(len, texts)))
    output = []
    for i, (row, bank) in enumerate(zip(rows, keys, strict=True)):
        shape = len(row["events"])
        affinity = queries[2*i:2*i+2] @ bank.T
        base = torch.tensor(row["gold_in_time"], dtype=torch.bool, device=affinity.device)
        assert len(base) == shape and int(base.sum()) == 1 and base[row["gold_event"]]
        modes = {}
        _, target = question_key(row["question"])
        for label, distance in (("clean", None), ("sham", 0), ("collision", 1)):
            if distance is None:
                compatible, ranked = base, affinity[:, :shape]
            else:
                added = parse_events([row[label]])[0]
                active = (target - added.start) % DAY < added.end - added.start
                assert active == (label == "collision") and row["scope"] != row["other_scope"]
                compatible = torch.cat((base, torch.tensor([active], device=base.device)))
                ranked = torch.cat((affinity[:, :shape], affinity[:, shape+distance:shape+distance+1]), -1)
            policies = {"dense": ranked[0], "resolved_clock_dense": ranked[1],
                        "compatible_time": ranked[1].masked_fill(~compatible, -torch.inf),
                        "compatible_time_and_scope": ranked[1, :shape].masked_fill(~base, -torch.inf)}
            # Place the added candidate in a fixed pseudo-random rank position
            # on exact ties without changing dense or date scores.
            n = len(policies["dense"])
            positions = torch.arange(n, device=base.device).roll(int(stable(row["id"])[:12], 16) % n)
            arms = {}
            for policy, scores in policies.items():
                order = positions[:len(scores)] if len(scores) == len(positions) else positions[positions < shape]
                ranked_slots = order[scores[order].argsort(descending=True, stable=True)[:2]].tolist()
                selected = ranked_slots[0]
                arms[policy] = {"support": selected == row["gold_event"], "decoy": selected == shape,
                                "support_top2": row["gold_event"] in ranked_slots and
                                                bool(torch.isfinite(scores[row["gold_event"]]))}
            modes[label] = arms
        output.append({k: row[k] for k in ("id", "scope", "difficulty", "kind")} | {"results": modes})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "fixture", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    rows = read(a.fixture)["rows"][a.shard::a.shards]
    result = evaluate(rows, Encoder(a.assets / "encoder", 96))
    save(a.output, {"fixture_sha256": digest(a.fixture), "models": read(a.assets / "revisions.json"),
                    "gpu": torch.cuda.get_device_name(), "elapsed_seconds": perf_counter() - start,
                    "rows": result})


if __name__ == "__main__":
    main()
