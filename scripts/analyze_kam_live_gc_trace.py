#!/usr/bin/env python3
"""Link Python collection pauses to exact B200 sentence-reuse replay tails."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median, quantiles

from analyze_kam_live_anchor_profile import evaluate
from analyze_kam_shared_serving import unpack

POLICIES = ("live_broad", "live_reuse")


def stats(values):
    return {"p50_ms": round(median(values), 6), "p95_ms": round(quantiles(values, n=20)[18], 6)}


def analyze(shards, fixture, archive):
    base = evaluate(shards, fixture, dict.fromkeys(POLICIES, archive))
    effects = {}
    for arm in POLICIES:
        collected, residual, by_generation, large = [], [], Counter(), []
        for shard in shards:
            for position, row in enumerate(shard["rows"]):
                request = row["timing_ms"][arm]
                events = row["records"][arm]["gc_events"]
                measured = sum(e["ms"] for e in events)
                assert 0 <= measured <= request
                collected.append(measured)
                residual.append(request - measured)
                for e in events:
                    assert e["generation"] in (0, 1, 2) and e["collected"] >= 0 and e["ms"] > 0
                    by_generation[e["generation"]] += 1
                    if e["generation"] == 2:
                        phase = row["phase_ms"][arm]
                        sampler_remainder = request - sum(phase.values())
                        assert sampler_remainder >= 0
                        qualifying = [k for k, v in (phase | {"other_or_initial": sampler_remainder}).items() if v >= e["ms"]]
                        large.append({"shard": shard["shard"], "query_ordinal": position, "order": row["order"],
                                      "pause_ms": round(e["ms"], 6), "collected_objects": e["collected"],
                                      "total_request_ms": round(request, 6), "possible_phases": qualifying,
                                      "rerank_ms": round(phase["remaining_rerank_ms"], 6)})
        effects[arm] = {"events_by_generation": dict(sorted(by_generation.items())),
                        "tracked_collection_ms": stats(collected),
                        "request_time_minus_tracked_pauses_diagnostic": stats(residual),
                        "generation_2_events": large}
    return {"paired_replay": base, "python_collection": effects}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", type=Path, nargs=4)
    for name in ("fixture", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    payload = analyze([unpack(f) for f in a.profiles], a.fixture, a.reference)
    a.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
