#!/usr/bin/env python3
"""Audit paired timestamp transformations on sealed external revision data."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_kam_archive_clock_encoding import split_inputs
from analyze_kam_external_clock_transfer import indexed
from analyze_kam_oecd_revisions import paired_quarter_seed

GEOMETRIES = ("dot", "radial")
RESULTS = ("joint", "numeric_exact")
OUTLIERS = ("future_outlier", "old_wrong_entity_outlier")
UNIT_PANELS = ("external", "external_clock_times_3", "external_raw_units")


def uniform_replay(current, baseline, archive_sections):
    assert current.keys() == baseline.keys()
    matched = Counter()
    for seed, source in current.items():
        old = baseline[seed]
        assert all(source[k] == old[k] for k in ("device", "steps", "schedule", "train_tokens", "banks"))
        assert all(source["archive_sha256"][k] == old["archive_sha256"][k] for k in archive_sections)
        common = source["results"].keys() & old["results"].keys()
        assert len(common) >= 4
        for key in sorted(common):
            a, b = [r["results"][key] for r in (source, old)]
            assert all(a[k] == b[k] for k in ("initial_sha256", "procedural"))
            assert all(a["archive"][k] == b["archive"][k] for k in archive_sections)
            matched[key] += 1
    return matched


def estimate(a, b, field, months=False, changed=None):
    assert a.keys() == b.keys() and a
    changes = {}
    for seed, group, culture, recode in a:
        key = seed, group, culture, recode
        if changed is not None and key not in changed:
            continue
        cluster = int(group.rsplit("-", 1)[1]) // 3
        block = (seed, f"{group}-{cluster}", culture, recode) if months else key
        changes[block] = int(b[key][field]) - int(a[key][field])
    return paired_quarter_seed(changes) if changes else None


def changed_rows(records, key, depth):
    changed = set()
    for seed, run in records.items():
        panels = run["results"]["radial/signed_log"]["archive"][key]
        for panel, selected in zip(panels, run["intervened_rows"][key], strict=True):
            assert len(selected) == len(panel["outcomes"])
            if panel["depth"] == depth:
                ids = ((seed, row["id"], row["period"], panel["recode"]) for row in panel["outcomes"])
                changed.update(k for k, v in zip(ids, selected, strict=True) if v)
    return changed


def summary(current, variants, panels, *, months=False, perturbation=False):
    result = {}
    for geo in GEOMETRIES:
        for depth in (16, 48):
            variants_data = {k: {panel: indexed(current, geo, k, panel, depth) for panel in panels} for k in variants}
            assert len({frozenset(table) for v in variants_data.values() for table in v.values()}) == 1
            test_counts = {k: changed_rows(current, k, depth) for k in OUTLIERS} if perturbation else {}
            schemes = {}
            for mode, arm in variants_data.items():
                scheme = {}
                for condition, rows in arm.items():
                    clean = arm["external"]
                    stamp = {"n": len(rows), "exact_records_vs_base_units": sum(row == clean[k] for k, row in rows.items()),
                             **{f"{field}_correct": sum(row[field] for row in rows.values()) for field in RESULTS}}
                    stamp["versus_clean_same_mode"] = {f: estimate(clean, rows, f, months=months) for f in RESULTS}
                    stamp["versus_signed_log_same_panel"] = {
                        f: estimate(variants_data["signed_log"][condition], rows, f, months=months) for f in RESULTS}
                    if condition in test_counts:
                        touched = test_counts[condition]
                        assert rows.keys() >= touched and all(rows[k] == v for k, v in clean.items() if k not in touched)
                        stamp["changed_records"] = len(touched)
                        stamp["versus_clean_changed_records"] = {
                            f: estimate(clean, rows, f, months=months, changed=touched) for f in RESULTS}
                    scheme[condition] = stamp
                schemes[mode] = scheme
            result[f"{geo}/{depth}"] = schemes
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("outliers", "monthly"))
    p.add_argument("--domain", choices=("archive_shaped", "mixed"), default="mixed")
    p.add_argument("--reference", nargs=3, type=Path, required=True)
    p.add_argument("--measurement", nargs=3, type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    runs, old = list(split_inputs(v, a.domain) for v in (a.measurement, a.reference))
    archives = ("earlier", "later")
    if a.mode == "outliers":
        archives += ("external",)
        variants = ("signed_log", "log_episode_mean", "log_episode_median", "log_available_mean")
        panels = ("external", *OUTLIERS, "external_clock_times_3")
        assert a.manifest is None and all(run.get("external_base_units", 1) == 1 for run in runs.values())
        provenance = {"source_fixture_sha256": next(iter(runs.values()))["archive_sha256"]["external"]}
    else:
        meta = json.loads(a.manifest.read_bytes())
        variants = ("linear", "signed_log", "log_episode_max", "log_episode_mean")
        panels = UNIT_PANELS
        assert all(run["archive_sha256"]["external"] == meta["fixture_sha256"] and run["external_base_units"] == 30
                   for run in runs.values())
        provenance = meta
    exact = uniform_replay(runs, old, archives)
    expected = {x["archive_sha256"]["external"] for x in runs.values()}
    assert len(expected) == 1
    values = summary(runs, variants, panels, months=a.mode == "monthly", perturbation=a.mode == "outliers")
    result = {"domain": a.domain, "provenance": provenance, "archived_fit_matches": exact, "results": values}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
