#!/usr/bin/env python3
"""Check date-coarsening parity and summarize safe admission and returned payloads."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_kam_guarded_fp32_sdpa import load


def summarize(measurements, old):
    origin = {r["seed"]: r for r in old}
    assert len(origin) == len(measurements) == 3 and set(origin) == {x["seed"] for x in measurements}
    by_panel = defaultdict(list)
    for row in measurements:
        control = origin[row["seed"]]
        assert all(row[key] == control[key] for key in ("source_sha256", "weights_sha256", "device"))
        assert row["device"] == "NVIDIA L4"
        for dataset, views in row["results"].items():
            for a, b in zip(views, control["results"][dataset], strict=True):
                assert {k: v for k, v in a.items() if k != "restricted"} == b
                by_panel[(dataset,a["depth"])].append(a)
    panels = {}
    for (title, depth), trials in by_panel.items():
        moods = defaultdict(Counter)
        conditions = ("all", "week_unique", "week_ambiguous")
        for run in trials:
            for i, q in enumerate(run["rows"]):
                prefix = "week_unique" if q["week_unique_snapshot"] else "week_ambiguous"
                assert q["week_unique_snapshot"] == (q["possibilities"] == 1)
                for view in ("all", prefix):
                    moods[view]["n"] += 1
                    for key, val in run["future_target_admitted"].items():
                        moods[(view, "future_target_admitted")][key] += val[i]
                    for name, values in run["predictions"].items():
                        key, value, future, target_future, unbound = values[i]
                        moods[(view, name)]["alias_correct"] += key
                        moods[(view, name)]["snapshot_correct"] += value
                        moods[(view, name)]["joint_exact"] += key and value
                        moods[(view, name)]["future_selected"] += future
                        moods[(view, name)]["target_future_selected"] += target_future
                        moods[(view, name)]["unbound"] += unbound
                        for p, choices in run["restricted"][name].items():
                            accepted, correct, extra = choices[i]
                            assert accepted or (not correct and not extra)
                            if name.endswith(("/day", "/week_end")):
                                assert not extra
                            stats = moods[(view, name, p)]
                            stats.update(accepted=accepted, joint_exact=correct, future_selected=extra)
        panels[f"{title}/{depth}"] = {view: {"n": moods[view]["n"],
                                               "future_target_admitted": moods[(view, "future_target_admitted")],
                                               "outputs": {tag: {"unrestricted": moods[(view, tag)], **{
                                                   p: moods[(view, tag, p)] for p in trials[0]["restricted"][tag]}}
                                                           for tag in trials[0]["predictions"]}}
                                     for view in conditions}
    return {"identical_daily_and_coarse_replays": True, "archive_inputs": measurements[0]["source_sha256"], "panels": panels}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, nargs=3, required=True)
    p.add_argument("--references", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    results = summarize(list(map(load,a.measurements)), list(map(load,a.references)))
    a.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
