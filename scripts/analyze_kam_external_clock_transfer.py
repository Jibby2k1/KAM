#!/usr/bin/env python3
"""Audit independent Dallas release transfer and clock unit changes."""

import argparse
import json
from pathlib import Path

from analyze_kam_archive_clock_encoding import split_inputs
from analyze_kam_oecd_revisions import paired_quarter_seed
from analyze_kam_temporal_clock_encoding import GEOMETRIES

MODES = ("linear", "signed_log")
DOMAINS = ("corrections", "archive_shaped", "mixed")
PANELS = ("external", "external_clock_times_3")


def replays(run, previous):
    assert {k: run["archive_sha256"][k] for k in ("earlier", "later")} == previous["archive_sha256"]
    assert all(run[k] == previous[k] for k in ("seed", "steps", "schedule", "banks"))
    for key, answer in run["results"].items():
        reference = previous["results"][key]
        assert answer["initial_sha256"] == reference["initial_sha256"] and answer["procedural"] == reference["procedural"]
        assert all(answer["archive"][era] == rows for era, rows in reference["archive"].items())


def indexed(records, geometry, mode, panel, depth):
    collection = {}
    for seed, run in records.items():
        source = run["results"][f"{geometry}/{mode}"]["archive"][panel]
        for block in source:
            if block["depth"] == depth:
                for r in block["outcomes"]:
                    key = seed, r["id"], r["period"], block["recode"]
                    assert key not in collection
                    collection[key] = r
    return collection


def summarize(raw):
    output = {}
    for d in (16, 48):
        indexed_cases = {}
        for geometry in GEOMETRIES:
            for mode in MODES:
                for domain, rows in raw.items():
                    for place in PANELS:
                        indexed_cases[geometry, mode, domain, place] = indexed(rows, geometry, mode, place, d)
        keys = next(iter(indexed_cases.values())).keys()
        assert len(keys) == 292 * 2 * 2 * 3 and all(table.keys() == keys for table in indexed_cases.values())
        effects = {}
        for arm in GEOMETRIES:
            for mode in MODES:
                for domain in DOMAINS:
                    a, b = [indexed_cases[arm, mode, domain, p] for p in PANELS]
                    effects[f"{arm}/{mode}/{domain}/clock_units"] = paired_quarter_seed({
                        k: int(b[k]["joint"]) - int(a[k]["joint"]) for k in keys})
            for place in PANELS:
                for domain in DOMAINS:
                    a, b = [indexed_cases[arm, m, domain, place] for m in MODES]
                    effects[f"{arm}/{domain}/{place}/signed_log"] = paired_quarter_seed({
                        k: int(b[k]["joint"]) - int(a[k]["joint"]) for k in keys})
                for mode in MODES:
                    a, b = [indexed_cases[arm, mode, domain, place] for domain in ("archive_shaped", "mixed")]
                    effects[f"{arm}/{mode}/{place}/mixed_vs_archive_shaped"] = paired_quarter_seed({
                        k: int(b[k]["joint"]) - int(a[k]["joint"]) for k in keys})
        records = {"/".join(k): {"q_recode_seed": len(r), "joint": sum(row["joint"] for row in r.values()),
                                  "numeric_exact": sum(row["numeric_exact"] for row in r.values())}
                   for k, r in indexed_cases.items()}
        output[str(d)] = {"counts": records, "paired": effects}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for domain in DOMAINS:
        for prefix in ("", "reference_"):
            option = prefix + domain
            p.add_argument("--" + option.replace("_", "-"), required=True, type=Path, nargs=3)
    p.add_argument("--source-provenance", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    provenance = json.loads(a.source_provenance.read_text())
    raw, runs = {}, 0
    for domain in DOMAINS:
        current = split_inputs(getattr(a, domain), domain)
        reference = split_inputs(getattr(a, "reference_"+domain), domain)
        assert current.keys() == reference.keys()
        for seed, run in current.items():
            assert run["archive_sha256"]["external"] == provenance["archive_fixture_sha256"]
            replays(run, reference[seed])
            for mode in run["results"].values():
                for base, rescaled in zip(*(mode["archive"][k] for k in PANELS), strict=True):
                    assert all(base[tag] == rescaled[tag] for tag in ("recode", "depth", "fixture_sha256"))
            runs += len(run["results"])
        raw[domain] = current
    payload = {"archive_fixture_sha256": provenance["archive_fixture_sha256"], "groups": provenance["counts"],
               "exact_preexisting_fits": runs, "results": summarize(raw)}
    a.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
