#!/usr/bin/env python3
"""Test whether earlier source calendars resolve coarsened revision dates."""

import argparse
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from hashlib import sha256
from itertools import combinations, permutations, product
from math import prod
import json
from pathlib import Path

from audit_kam_revision_clock_resolution import MODES, bucket, feasible_latest, payload, source_catalog

PRIORS = ("uniform", "weekday", "weekday_monthday")
CLOCKS = ("week", "fortnight", "month")


def calendar(archive):
    begin = date(2021, 1, 1).toordinal()
    stop = date(2023, 1, 1).toordinal()
    exposures = [date.fromordinal(d) for d in range(begin, stop)]
    releases = {date.fromordinal(d) for history in archive.values() for d, _ in history if begin <= d < stop}
    assert releases
    rates = []
    for feature in (date.weekday, lambda d: d.day):
        counts = Counter(map(feature, releases))
        support = Counter(map(feature, exposures))
        rates.append({name: (counts[name]+1)/(count+2) for name, count in support.items()})
    return rates, len(releases)


def coefficients(weights, maximum):
    """Elementary symmetric polynomials for weighted distinct day assignments."""
    coeff = [1.0] + [0.0]*maximum
    for j, w in enumerate(weights, 1):
        for k in range(min(j, maximum), 0, -1):
            coeff[k] += w * coeff[k-1]
    return coeff


def posterior(previous, current, cutoff, limits, weight, ordered):
    low, high = limits
    n = len(current)
    before = coefficients([weight(day) for day in range(low, cutoff+1)], n)
    after = coefficients([weight(day) for day in range(cutoff+1, high+1)], n)
    mass = Counter()
    for k, (a, b) in enumerate(zip(before, reversed(after), strict=True)):
        if a*b == 0:
            continue
        possible = (current[k-1:k] if ordered else current) if k else (previous[-1:] if ordered else previous)
        choices = possible or [None]
        for candidate in choices:
            mass[candidate] += a*b/len(choices)
    norm = sum(mass.values())
    assert norm > 0
    return {i: p/norm for i, p in mass.items()}


def validate_posterior():
    cases = 0
    weight = lambda d: float(d*d + 2)
    for m, n, cutoff, ordered in product(range(3), range(5), range(1, 5), (False, True)):
        earlier = [-(j+1) for j in range(m)]
        current = list(range(n))
        mass = Counter()
        for selected in combinations(range(1, 5), n):
            options = [current] if ordered else list(permutations(current))
            for assignment in options:
                candidates = [record for d, record in zip(selected, assignment, strict=True) if d <= cutoff]
                preceding = earlier[-1:] if ordered else earlier
                for item in candidates[-1:] or preceding or [None]:
                    mass[item] += prod(map(weight, selected)) / len(options) / (1 if candidates else max(1, len(preceding)))
        reference = {k: v / sum(mass.values()) for k, v in mass.items()}
        got = posterior(earlier, current, cutoff, (1, 4), weight, ordered)
        assert got.keys() == reference.keys() and all(abs(got[k]-v) < 1e-12 for k, v in reference.items())
        cases += 1
    return cases


def distributions(history, cutoff, clock, order, weekday, monthday):
    limits = bucket(cutoff, clock)
    previous, current = [], []
    newest = 0
    for day, opaque in history:
        start, _ = bucket(day, clock)
        if start < limits[0] and start >= newest:
            if start > newest:
                newest, previous = start, []
            previous.append(opaque)
        elif start == limits[0]:
            current.append(opaque)
    feasible = feasible_latest(history, cutoff, lambda day: bucket(day, clock), ordered=order)
    expected = {r[1] if r else None for r in feasible}
    options = {}
    for name, weight in (
        ("uniform", lambda _: 1.0),
        ("weekday", lambda d: weekday[date.fromordinal(d).weekday()]),
        ("weekday_monthday", lambda d: weekday[date.fromordinal(d).weekday()] * monthday[date.fromordinal(d).day]),
    ):
        p = posterior(previous, current, cutoff, limits, weight, order)
        assert set(p) == expected, (clock, name, order, limits, cutoff, len(previous), len(current), len(p), len(expected))
        assert abs(sum(p.values())-1) < 1e-10
        options[name] = p
    return options


def cutoffs(questions, daily):
    seen = set()
    for q in questions:
        if q["gold"] is None:
            continue
        when = q["cutoff"]
        for cutoff in range(when-7, when+8) if daily else [when]:
            key = (q["period"], cutoff)
            if key in seen or date.fromordinal(cutoff).year != date.fromordinal(when).year:
                continue
            seen.add(key)
            yield cutoff, q


def predictions(path, catalog, source_digest, rates, later, daily=False):
    raw = payload(path)
    fixture = json.loads(raw)
    assert fixture["source"]["source_sha256"] == source_digest
    for group in fixture["rows"]["test"]:
        key = group["series"], group["observation"]
        original = catalog[key]
        # Only the coarse day interval and optionally the chronological order are observed.
        mapping = {d: sha256(json.dumps([*key, d], separators=(",", ":")).encode()).hexdigest() for d, _ in original}
        numeric = {mapping[d]: number for d, number in original} | {None: None}
        history = [(d, mapping[d]) for d, _ in original]
        for cutoff, q in cutoffs(group["questions"], daily):
            when = date.fromordinal(cutoff)
            if when.year < 2023:
                continue
            stage = "later" if later else ("development" if when.year == 2023 else "original_test")
            assert when.year >= (2025 if later else 2023)
            target = group["candidates"][q["gold"]]
            assert key == (target["series"], target["observation"])
            day, number = max(((d, v) for d, v in original if d <= cutoff), default=(None, None))
            if cutoff == q["cutoff"]:
                assert day == target["vintage"] and number == Decimal(target["value"])
            truth = mapping.get(day)
            grade = {"stage": stage, "question": group["id"], "period": q["period"], "cutoff": when.isoformat(),
                     "has_release": day is not None,
                     "quarter": f"{when.year}-Q{(when.month-1)//3+1}", "outcomes": {}}
            for clock, mode in product(CLOCKS, MODES):
                found = distributions(history, cutoff, clock, mode == "bucket_and_order", *rates)
                keys = found["uniform"].keys()
                assert truth in keys
                row = {"alternatives": len(keys), "numeric_alternatives": len({numeric[x] for x in keys}), "policies": {}}
                for name, p in found.items():
                    best = min(p, key=lambda k: (-round(p[k], 12), k or ""))
                    numeric_mass = Counter()
                    for k, value in p.items():
                        numeric_mass[numeric[k]] += value
                    numer = min(numeric_mass, key=lambda k: (-round(numeric_mass[k], 12), str(k)))
                    assert max(p.values()) - p[best] < 1e-11
                    row["policies"][name] = {"confidence": round(p[best], 12), "strict_exact": best == truth,
                                              "numeric_exact": numer == numeric[truth], "truth_probability": round(p[truth], 12)}
                grade["outcomes"][f"{clock}/{mode}"] = row
            yield grade


def select_threshold(rows, target=.95, per_quarter=False):
    by_confidence = defaultdict(Counter)
    for label, row in rows:
        key = label if per_quarter else "all"
        by_confidence[row["confidence"]].update({(key, "n"): 1, (key, "correct"): row["strict_exact"]})
    candidates = [(0, 0, 1.01)]
    counts = Counter()
    for prob in sorted(by_confidence, reverse=True):
        counts.update(by_confidence[prob])
        blocks = [(n, counts[key, "correct"]) for (key, name), n in counts.items() if name == "n"]
        if all(correct >= target*n for n, correct in blocks):
            candidates.append((sum(n for n, _ in blocks), sum(c for _, c in blocks), prob))
    return max(candidates)[2]


def summary(rows, primary=None, per_quarter=False):
    by_stage = {name: [row for row in rows if row["stage"] == name] for name in ("development", "original_test", "later")}
    outcomes = {}
    for case in rows[0]["outcomes"]:
        panels = {name: [row["outcomes"][case] for row in source] for name, source in by_stage.items()}
        fit = [(rec["quarter"], r) for rec, r in zip(by_stage["development"], panels["development"], strict=True)
               if r["alternatives"] > 1]
        limits = ({p: primary[case]["development"]["policies"][p]["decline_threshold"] for p in PRIORS}
                  if primary else {p: select_threshold([(quarter, r["policies"][p]) for quarter, r in fit],
                                                       per_quarter=per_quarter) for p in PRIORS})
        reported = {}
        for stage, panel in panels.items():
            ambiguous = [r for r in panel if r["alternatives"] > 1]
            row = {"n": len(panel), "strict_unique": len(panel)-len(ambiguous), "quarters": sorted({r["quarter"] for r in by_stage[stage]}),
                   "numeric_unique": sum(r["numeric_alternatives"] == 1 for r in panel), "policies": {}}
            for name in PRIORS:
                selected = [r["policies"][name] for r in ambiguous]
                accepted = [r for r in selected if r["confidence"] >= limits[name]]
                row["policies"][name] = {"correct_on_ambiguous": sum(r["strict_exact"] for r in selected),
                                         "ambiguous_numeric_correct": sum(r["numeric_exact"] for r in selected),
                                         "decline_threshold": limits[name], "accepted_ambiguous": len(accepted),
                                         "accepted_ambiguous_correct": sum(r["strict_exact"] for r in accepted)}
                quarter_counts = defaultdict(Counter)
                for item, result in zip(by_stage[stage], panel, strict=True):
                    if result["alternatives"] == 1:
                        continue
                    branch = result["policies"][name]
                    take = branch["confidence"] >= limits[name]
                    quarter_counts[item["quarter"]].update(n=1, accepted=take, correct=take and branch["strict_exact"])
                row["policies"][name]["ambiguity_by_quarter"] = quarter_counts
            reported[stage] = row
        outcomes[case] = reported
    return outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("unemployment", "industry"):
        for mode in ("source", "original", "later"):
            p.add_argument("--" + field + "-" + mode, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--daily-probe", action="store_true")
    a = p.parse_args()
    result = {"exhaustive_posterior_checks": validate_posterior()}
    for name in ("unemployment", "industry"):
        archive, sha, series = source_catalog(getattr(a, name+"_source"), name)
        rates, n = calendar(archive)
        inputs = tuple(getattr(a, f"{name}_{stage}") for stage in ("original", "later"))
        rows = [row for i, source in enumerate(inputs) for row in predictions(source, archive, sha, rates, bool(i))]
        result[name] = {"source_sha256": sha, "series": series, "fit_release_days": n, "weekday_rates": rates[0],
                        "monthday_rates": rates[1], "summary": summary(rows), "rows": rows}
        if a.daily_probe:
            daily = [row for i, source in enumerate(inputs) for row in predictions(source, archive, sha, rates, bool(i), daily=True)]
            result[name]["daily"] = {"summary": summary(daily, result[name]["summary"]),
                                     "matched_calibration": summary(daily),
                                     "per_quarter_calibration": summary(daily, per_quarter=True), "rows": daily}
    a.output.write_text(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
