#!/usr/bin/env python3
"""Verify matched cutoff training and compare real, unmodified revision dates."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_clock_archive_controls import estimate
from analyze_kam_native_musique import compare

MODES = ("reference", "all_fact_clocks", "wrong_address_only")
TYPED_MODES = ("reference", "address_rank", "address_rank_sham")
ADDITIVE_MODES = ("reference", "fact_rank_feature", "alias_fact_rank_feature", "alias_fact_sham")
GEOMETRIES = ("dot", "radial")
METRICS = ("joint", "numeric_exact")


def read(path):
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)


def lags(path):
    rows = read(path)["rows"]["test"]
    values = {}
    for group in rows:
        candidates = group["candidates"]
        for question in group["questions"]:
            tag = group["id"], question["period"]
            assert tag not in values
            record = candidates[question["gold"]]
            assert record["observation"] == group["observation"] and record["series"] == group["series"]
            delta = question["cutoff"] - record["vintage"]
            assert delta >= 0
            values[tag] = delta
    return values


def indexed(runs, geometry, arm, source, depth):
    indexed = {}
    for row in runs:
        for panel in row["arms"][f"{geometry}/{arm}"]["archives"][source]:
            if panel["depth"] != depth:
                continue
            for case in panel["outcomes"]:
                key = row["seed"], case["id"], case["period"], panel["recode"]
                assert key not in indexed
                indexed[key] = case
    return indexed


def measured(a, b, time_lag=None):
    assert a.keys() == b.keys()
    output = {"n": len(a), "counts": {field: sum(v[field] for v in b.values()) for field in METRICS},
              "vs_reference": {field: estimate(a, b, field, months=time_lag is not None) for field in METRICS}}
    if time_lag:
        strata = {"release_day": set(), "one_to_fourteen_days": set(), "over_fourteen_days": set()}
        for key in a:
            _, id_, period, _ = key
            age = time_lag[id_, period]
            tag = "release_day" if age == 0 else ("one_to_fourteen_days" if age <= 14 else "over_fourteen_days")
            strata[tag].add(key)
        output["cutoff_lag"] = {
            tag: {"n": len(subset), "joint": sum(b[k]["joint"] for k in subset),
                  "paired_joint": estimate(a, b, "joint", months=True, changed=subset)}
            for tag, subset in strata.items() if subset}
    return output


def synthetic(runs, geometry, modes):
    frames = {}
    for case in ("original", "corrections"):
        depths = {}
        for d in ("16", "48", "128", "512"):
            metrics = {m: {} for m in modes}
            for run in runs:
                arms = [run["arms"][f"{geometry}/{m}"]["procedural"][case][d] for m in modes]
                assert len({p["fixture_sha256"] for p in arms}) == 1
                truth = list(map(all, arms[0]["outcomes"]))
                assert len(truth) == 512
                for mode, panel in zip(modes, arms, strict=True):
                    values = list(map(all, panel["outcomes"]))
                    metrics[mode][run["seed"]] = {"n": len(values), "joint": sum(values),
                                                  "vs_reference": compare(values, truth, [[i] for i in range(len(values))])}
            depths[d] = metrics
        frames[case] = depths
    return frames


def audit_routes(runs, geometry, mode, source, depth):
    counts = {}
    for run in runs:
        arm = run["arms"][f"{geometry}/{mode}"]
        for trained, diagnostic in zip(arm["archives"][source], arm["route_diagnostics"][source], strict=True):
            assert all(trained[k] == diagnostic[k] for k in ("depth", "recode", "fixture_sha256"))
            if trained["depth"] != depth:
                continue
            for measured, choices in zip(trained["outcomes"], diagnostic["outcomes"], strict=True):
                assert all(choices["learned"][:2]) == measured["joint"] and not choices["learned"][2]
                for name, (key, value, abstain) in choices.items():
                    counts.setdefault(name, Counter()).update({"n": 1, "key": key, "payload": value,
                                                                 "joint": key and value, "abstention": abstain})
    return counts


def compare_sources(runs, geometry, phase, modes, additive=False):
    result = {}
    for source, lag in phase.items():
        panels = {}
        for d in (16, 48):
            fields = {m: indexed(runs, geometry, m, source, d) for m in modes}
            assert len(fields["reference"]) == (len(lag) * 6 if lag is not None else 3504)
            for row in runs:
                panels_in = [[p["fixture_sha256"] for p in row["arms"][f"{geometry}/{m}"]["archives"][source]]
                             for m in modes]
                assert all(x == panels_in[0] for x in panels_in[1:])
            panels[str(d)] = {m: measured(fields["reference"], fields[m], lag) for m in modes}
            if additive:
                for m in modes:
                    panels[str(d)][m]["route_diagnostics"] = audit_routes(runs, geometry, m, source, d)
        result[source] = panels
    return result


def analyze(runs, sources):
    assert len(runs) == len({r["seed"] for r in runs}) == 3
    assert {r["device"] for r in runs} == {"NVIDIA L4"}
    assert all((r["steps"], r["tokens"], r["schedule"], r["clock"]) ==
               (3200, 10556800, [[16, 100], [48, 34]], "log_episode_mean") for r in runs)
    meta = {k: sha256(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()).hexdigest()
            for k, p in sources.items()}
    audit = {}
    for run in runs:
        assert all(run["fixture_hashes"][k] == v for k, v in meta.items())
        note = run["reference_audit"]
        assert set(note) == set(GEOMETRIES)
        for row in note.values():
            assert row["unemployment_replay"] and row["max_loss_difference"] == 0
            assert len(row["procedural_outcomes_replayed"]) == 8 and set(row["procedural_outcomes_replayed"].values()) == {512}
        audit[run["seed"]] = note
    assert len({r["training_domain"] for r in runs}) == 1
    fractions = runs[0].get("phase_mixture_fractions", {})
    assert all(r.get("phase_mixture_fractions", {}) == fractions for r in runs)
    typed = runs[0].get("typed_local_clock", False)
    assert all(r.get("typed_local_clock", False) == typed for r in runs)
    additive = runs[0].get("additive_local_clocks", False)
    assert all(r.get("additive_local_clocks", False) == additive for r in runs)
    if fractions:
        modes = ("reference", *sorted(fractions, key=fractions.get))
    elif typed:
        modes = TYPED_MODES
    elif additive:
        modes = ADDITIVE_MODES
    else:
        modes = MODES
    assert all(set(r["arms"]) == {f"{g}/{m}" for g in GEOMETRIES for m in modes} for r in runs)
    phase = {"dallas": None} | {k: lags(v) for k, v in sources.items()}
    outcomes = {}
    for geo in GEOMETRIES:
        outcomes[geo] = {"procedural": synthetic(runs, geo, modes), "real": compare_sources(runs, geo, phase, modes, additive)}
    return {"domain": runs[0]["training_domain"], "fixture_hashes": runs[0]["fixture_hashes"],
            "phase_mixture_fractions": fractions, "typed_local_clock": typed,
            "cutoff_lag_counts": {k: dict(Counter(p.values())) for k, p in phase.items() if p}, "reference_audit": audit,
            "outcomes": outcomes} | ({"additive_local_clocks": True} if additive else {})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("unemployment", "industry", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("later-unemployment", "later-industry"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--measurement", nargs=3, type=Path, required=True)
    a = p.parse_args()
    runs = sorted(map(read, a.measurement), key=lambda r: r["seed"])
    trials = analyze(runs, {k: getattr(a, k) for k in ("unemployment", "industry", "later_unemployment", "later_industry")
                            if getattr(a, k)})
    a.output.write_text(json.dumps(trials, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
