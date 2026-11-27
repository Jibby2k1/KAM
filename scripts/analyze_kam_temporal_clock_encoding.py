#!/usr/bin/env python3
"""Compare clock feature transformations on matched correction and archive tests."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean

from analyze_kam_native_musique import compare

MODES = ("linear", "signed_log", "signed_log_guard")
GEOMETRIES = ("dot", "radial")
PAIRS = (("signed_log", "linear"), ("signed_log_guard", "signed_log"))
COMPACT_MODES = ("signed_log", "signed_log_guard", "signed_log_compact")
COMPACT_PAIRS = (("signed_log_compact", "signed_log"), ("signed_log_compact", "signed_log_guard"))


def read_measurement(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def panel(measurements, clusters, pairs):
    return {"n": len(next(iter(measurements.values()))), "counts": {k: sum(v) for k, v in measurements.items()},
            "difference": {f"{a}_vs_{b}": compare(measurements[a], measurements[b], clusters) for a, b in pairs}}


def summarize(seed, modes, pairs):
    result = {}
    network = seed["results"]
    assert len({q["initial_sha256"] for q in network.values()}) == 1
    for geometry in GEOMETRIES:
        got = {m: network[f"{geometry}/{m}"] for m in modes}
        procedures = {}
        for domain in ("original", "corrections"):
            procedures[domain] = {}
            for depth in ("16", "48", "128", "512"):
                record = {m: item["procedural"][domain][depth] for m, item in got.items()}
                assert len({r["fixture_sha256"] for r in record.values()}) == 1
                outcomes = {m: [all(x) for x in r["outcomes"]] for m, r in record.items()}
                procedures[domain][depth] = panel(outcomes, [[i] for i in range(len(outcomes[modes[0]]))], pairs)
        archive = {}
        for era in ("earlier", "later"):
            archive[era] = {}
            for depth in (16, 48):
                panels = {m: [row for row in r["archive"][era] if row["depth"] == depth] for m, r in got.items()}
                assert len({tuple(z["fixture_sha256"] for z in v) for v in panels.values()}) == 1
                all_rows = {m: [record for r in value for record in r["outcomes"]] for m, value in panels.items()}
                check = lambda r: (r["id"], r["period"], r["series"])
                assert all([check(q) for q in r] == [check(q) for q in all_rows[modes[0]]] for r in all_rows.values())
                groups = defaultdict(list)
                for j, r in enumerate(all_rows[modes[0]]):
                    groups[int(r["id"].rsplit("-", 1)[1])].append(j)
                archive[era][str(depth)] = {k: panel({m: [r[k] for r in rows] for m, rows in all_rows.items()},
                                                    list(groups.values()), pairs) for k in ("joint", "numeric_exact")}
        result[geometry] = {"procedures": procedures, "archive": archive,
                            "training_seconds": {m: item["fit"]["seconds"] for m, item in got.items()}}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, nargs=3, required=True)
    p.add_argument("--compaction-controls", type=Path, nargs=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    measured = sorted((read_measurement(f) for f in a.measurements), key=lambda r: r["seed"])
    assert all(r["device"] == "NVIDIA L4" for r in measured) and len({r["seed"] for r in measured}) == 3
    assert all(r["steps"] == measured[0]["steps"] and r["schedule"] == measured[0]["schedule"] and
               r["archive_sha256"] == measured[0]["archive_sha256"] for r in measured)
    modes, pairs = (COMPACT_MODES, COMPACT_PAIRS) if a.compaction_controls else (MODES, PAIRS)
    parity = {}
    if a.compaction_controls:
        prior = sorted((read_measurement(f) for f in a.compaction_controls), key=lambda r: r["seed"])
        for old, new in zip(prior, measured, strict=True):
            assert all(old[k] == new[k] for k in ("seed", "steps", "schedule", "banks", "archive_sha256"))
            known = [f"{g}/{m}" for g in GEOMETRIES for m in COMPACT_MODES[:2]]
            parity[str(new["seed"])] = {mode: all(old["results"][mode][k] == new["results"][mode][k]
                                                 for k in ("procedural", "archive")) for mode in known}
    seeds = {str(r["seed"]): summarize(r, modes, pairs) for r in measured}
    header = {}
    for g in GEOMETRIES:
        runs = [seed[g] for seed in seeds.values()]
        panels = {
            "corrected_48": [r["procedures"]["corrections"]["48"] for r in runs],
            "corrected_512": [r["procedures"]["corrections"]["512"] for r in runs],
            "earlier_archive_48": [r["archive"]["earlier"]["48"]["joint"] for r in runs],
            "later_archive_48": [r["archive"]["later"]["48"]["joint"] for r in runs]}
        header[g] = {k: {m: mean(r["counts"][m] / r["n"] for r in rows) for m in modes} for k, rows in panels.items()}
    result = {"steps": measured[0]["steps"], "archive_sha256": measured[0]["archive_sha256"],
              "header": header, "seeds": seeds}
    if parity:
        result["prior_outcome_parity"] = parity
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
