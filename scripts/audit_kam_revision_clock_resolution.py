#!/usr/bin/env python3
"""Identify all exact as-of snapshots consistent with coarsened publication dates."""

import argparse
from calendar import monthrange
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from itertools import permutations, product
import gzip
from hashlib import sha256
import json
from pathlib import Path

from prepare_kam_eurostat_vintages import INDICATORS, records

CLOCKS = ("day", "week", "fortnight", "month")
MODES = ("bucket_only", "bucket_and_order")


def bucket(day, clock):
    if clock == "month":
        d = date.fromordinal(day)
        start = day - d.day + 1
        return start, start + monthrange(d.year, d.month)[1] - 1
    width = {"day": 1, "week": 7, "fortnight": 14}[clock]
    start = ((day - 1) // width) * width + 1
    return start, start + width - 1


def feasible_latest(history, cutoff, resolution, *, ordered):
    """Each record has an ID and a date interval shared with its bucket peers."""
    lo, hi = resolution(cutoff)
    preceding = []
    spanning = []
    previous_lo = -1
    for day, payload in history:
        start, end = resolution(day)
        assert start <= day <= end
        if start < lo:
            if start > previous_lo:
                previous_lo, preceding = start, []
            if start == previous_lo:
                preceding.append((day, payload))
        elif start == lo:
            spanning.append((day, payload))
    n = len(spanning)
    assert n <= hi - lo + 1
    before_min = max(0, n - (hi - cutoff))
    before_max = min(n, cutoff - lo + 1)
    values = set()
    if before_min == 0:
        if not preceding:
            values.add(None)
        else:
            values.update(preceding[-1:] if ordered else preceding)
    if before_max:
        values.update(spanning[max(before_min, 1)-1:before_max] if ordered else spanning)
    return values


def brute_force():
    for prior_n, curr_n, later_n in product(range(4), repeat=3):
        history = [(lo + i, str(i + lo)) for lo, count in ((1, prior_n), (4, curr_n), (7, later_n))
                   for i in range(count)]
        subsets = ([r for r in history if low <= r[0] <= low+2] for low in (1, 4, 7))
        fixed = list(subsets)
        for ordered, cutoff in product((False, True), (4, 5, 6)):
            assignments = [list(permutations(range(low, low+3), len(rows)))
                           for low, rows in zip((1, 4, 7), fixed, strict=True)]
            if ordered:
                assignments = [[p for p in block if tuple(sorted(p)) == p] for block in assignments]
            checked = set()
            for combo in product(*assignments):
                marked = [(d, item) for days, rows in zip(combo, fixed, strict=True)
                          for d, item in zip(days, rows, strict=True) if d <= cutoff]
                checked.add(max(marked)[1] if marked else None)
            coarse = lambda x: (1 + ((x-1)//3)*3, 3 + ((x-1)//3)*3)
            assert checked == feasible_latest(history, cutoff, coarse, ordered=ordered)
    return 128 * 3


def payload(data):
    raw = data.read_bytes()
    return gzip.decompress(raw) if data.suffix == ".gz" else raw


def source_catalog(path, indicator):
    raw = payload(path)
    code, geos, dims, fixed = INDICATORS[indicator]
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        sealed = Path(directory) / "source.json"
        sealed.write_bytes(raw)
        _, catalog = records(sealed, geos, dims, fixed)
    grouped = defaultdict(list)
    for series, obs, day, number in catalog:
        grouped[series, obs].append((day, Decimal(number)))
    for key, rows in grouped.items():
        rows.sort()
        assert len(rows) == len({row[0] for row in rows}), key
    return grouped, sha256(raw).hexdigest(), code


def assess(fixture_path, archive, input_hash):
    content = payload(fixture_path)
    fixture = json.loads(content)
    assert fixture["source"]["source_sha256"] == input_hash
    cuts = fixture["source"]["days_after_month_end"]
    outcomes = []
    hits = {"candidate_pool": defaultdict(Counter), "all_source": defaultdict(Counter)}
    for group in fixture["rows"]["test"]:
        key = group["series"], group["observation"]
        all_rows = archive[key]
        candidates = group["candidates"]
        selected = sorted((r["vintage"], Decimal(r["value"])) for r in candidates
                          if (r["series"], r["observation"]) == key)
        assert set(selected).issubset(all_rows)
        for q in group["questions"]:
            if q["gold"] is None:
                continue
            gold_row = candidates[q["gold"]]
            target = gold_row["vintage"], Decimal(gold_row["value"])
            cutoff = q["cutoff"]
            assert target == max(r for r in all_rows if r[0] <= cutoff)
            assert target == max(r for r in selected if r[0] <= cutoff)
            row = {"id": group["id"], "period": q["period"], "results": {}}
            for name, history in (("candidate_pool", selected), ("all_source", all_rows)):
                row["results"][name] = {}
                for clock, mode in product(CLOCKS, MODES):
                    result = feasible_latest(history, cutoff, lambda day: bucket(day, clock),
                                             ordered=mode == "bucket_and_order")
                    assert target in result
                    numbers = {x[1] if x else None for x in result}
                    admitted = {"strict_unique": len(result) == 1, "numeric_unique": len(numbers) == 1,
                                "could_be_unavailable": None in result, "alternatives": len(result)}
                    key_ = f"{clock}/{mode}"
                    row["results"][name][key_] = admitted
                    stats = hits[name][key_]
                    stats.update({"n": 1, **admitted})
            outcomes.append(row)
    for by_scope in hits.values():
        for m in MODES:
            assert by_scope[f"day/{m}"]["n"] == by_scope[f"day/{m}"]["strict_unique"] == len(outcomes)
    return {"fixture_sha256": sha256(content).hexdigest(), "cutoffs": cuts, "counts": hits, "rows": outcomes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unemployment-source", type=Path, required=True)
    p.add_argument("--industry-source", type=Path, required=True)
    for panel in ("unemployment", "industry", "later-unemployment", "later-industry"):
        p.add_argument("--" + panel, required=True, type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    results = {"brute_force_cases": brute_force(), "panels": {}}
    for indicator in ("unemployment", "industry"):
        catalog, stamp, code = source_catalog(getattr(a, indicator + "_source"), indicator)
        for panel in (indicator, "later_" + indicator):
            results["panels"][panel] = {"source": code, "source_sha256": stamp} | assess(getattr(a, panel), catalog, stamp)
    a.output.write_text(json.dumps(results, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
