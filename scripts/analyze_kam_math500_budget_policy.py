#!/usr/bin/env python3
"""Compare fixed stopping rules on the same recorded MATH-500 rollouts."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import ARMS, SEEDS, key
from kam_math_kl_protocol import BATCH_SIZE, test_runs, write_json
from summarize_kam_adapter_preservation import difference, read_archive


SIZES = (1, 8, BATCH_SIZE)


def measure(rows, early_repetitive):
    n = len(rows)
    outputs = np.array([row["output_tokens"] for row in rows])
    full_steps = outputs + np.array([row["stop"] == "eos" for row in rows])
    full_ok = np.array([row["full"]["correct"] for row in rows])
    prefix_ok = np.array([row["prefix"]["correct"] for row in rows])
    prefix_parse = np.array([row["prefix"]["parsed"] for row in rows])
    order = sorted(range(n), key=lambda i: (rows[i]["prompt_tokens"], i))
    batches = {size: [order[start:start+size] for start in range(0, n, size)] for size in SIZES}

    def stats(steps, correct):
        return {
            "score": float(correct.mean()),
            "decoder_steps": sum(int(steps[group].max()) for group in batches[BATCH_SIZE]),
            "batched_token_slots": {str(size): sum(len(group) * int(steps[group].max()) for group in groups)
                                    for size, groups in batches.items()},
            "correct_gained_vs_384": int((correct & ~prefix_ok).sum()),
            "correct_lost_vs_1024": int((full_ok & ~correct).sum()),
        }

    hold_at_prefix = (full_steps > 384) & prefix_parse
    flexible_steps = np.where(hold_at_prefix, 384, full_steps)
    flexible_ok = np.where(hold_at_prefix, prefix_ok, full_ok)
    stop_on_repeat = (full_steps > 384) & np.asarray([row["id"] in early_repetitive for row in rows])
    repetition_steps = np.where(stop_on_repeat, 384, full_steps)
    repetition_ok = np.where(stop_on_repeat, prefix_ok, full_ok)
    return {"short": stats(np.minimum(full_steps, 384), prefix_ok), "full": stats(full_steps, full_ok),
            "extend_unparsed": {**stats(flexible_steps, flexible_ok),
                                "stopped_with_candidate": int(hold_at_prefix.sum())},
            "stop_repetitive": {**stats(repetition_steps, repetition_ok),
                                "stopped_for_repetition": int(stop_on_repeat.sum())}}, (prefix_ok, full_ok, flexible_ok,
                                                                                      repetition_ok)


def summarize(archive, selection, repetition):
    read = read_archive(archive)
    registered = json.loads(selection.read_text())
    repeat = json.loads(repetition.read_text())
    assert repeat["model"] == registered["model"]
    recipe = registered["selected_recipe"]
    stats, flags = {}, {}
    for run in test_runs(recipe):
        name = key(*run)
        result = read(f"graded-{name}")
        assert result["metadata"]["model"] == registered["model"]
        stats[name], flags[name] = measure(result["records"], set(repeat["flag_ids"][name]["prefix"]))
    contrasts = {}
    for arm in ARMS:
        short, full, flexible, repetition_limited = (np.stack(items) for items in
                                 zip(*(flags[key(seed, recipe, arm)] for seed in SEEDS), strict=True))
        contrasts[arm] = {"conditional_minus_short": difference(flexible, short),
                          "conditional_minus_full": difference(flexible, full),
                          "repeat_minus_full": difference(repetition_limited, full)}
    return {"model": registered["model"], "recipe": recipe, "checkpoints": stats, "contrasts": contrasts}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("selection", type=Path)
    cli.add_argument("repetition", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    write_json(args.output, summarize(args.archive, args.selection, args.repetition))
