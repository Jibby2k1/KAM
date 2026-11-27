#!/usr/bin/env python3
"""Seal exact as-of lookup cohorts from Eurostat's monthly unemployment vintages."""

import argparse
from bisect import bisect_right
from calendar import monthrange
from collections import Counter, defaultdict
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
INDICATORS = {
    "unemployment": (
        "ei_lm_m_vtg", ("DE", "ES", "FR", "IT", "NL", "PL"),
        ("freq", "unit", "s_adj", "sex", "age", "geo", "revdate", "time"),
        {"freq": "M", "unit": "PC_ACT", "s_adj": "SA", "sex": "T", "age": "TOTAL"},
    ),
    "industry": (
        "ei_is_m_vtg", ("AT", "BE", "CZ", "DK", "FI", "SE"),
        ("freq", "revdate", "s_adj", "nace_r2", "unit", "geo", "time"),
        {"freq": "M", "unit": "RT1", "s_adj": "SCA", "nace_r2": "B-D"},
    ),
}
YEARS = (2021, 2024)
CUTOFFS = (90, 365)


def month_id(month):
    assert len(month) == 7
    y, m = map(int, month.split("-"))
    assert 1 <= m <= 12
    return y * 12 + m - 1


def records(source, geos, dimensions, fixed):
    payload = json.loads(source.read_bytes())
    assert tuple(payload["id"]) == dimensions
    shapes = payload["size"]
    categories = []
    for key, size in zip(payload["id"], shapes, strict=True):
        indexed = payload["dimension"][key]["category"]["index"]
        assert set(indexed.values()) == set(range(size))
        categories.append(sorted(indexed, key=indexed.get))
    assert all(categories[dimensions.index(key)] == [value] for key, value in fixed.items())
    assert set(categories[dimensions.index("geo")]) == set(geos)
    results = []
    for loc, value in payload["value"].items():
        assert type(value) in (float, int)
        ix = int(loc)
        indices = []
        for dim in reversed(shapes):
            ix, rem = divmod(ix, dim)
            indices.append(rem)
        assert ix == 0
        selections = {k: labels[j] for k, labels, j in zip(dimensions, categories, reversed(indices), strict=True)}
        country, day, month = [selections[k] for k in ("geo", "revdate", "time")]
        results.append((country, month_id(month), date.fromisoformat(day).toordinal(), str(value)))
    return payload["updated"], results


def choose_donors(by_month, geos, geo, month, cutoffs):
    common, foreign = [], []
    for neighbor in range(month - 5, month + 6):
        for side in geos:
            if (side, neighbor) == (geo, month):
                continue
            collection = common if geo == side else foreign
            for row in by_month[side, neighbor]:
                if min(abs(row["vintage"] - cutoff) for cutoff in cutoffs) <= 200:
                    collection.append(row)
    def score(row):
        return (min(abs(row["vintage"] - t) for t in cutoffs), abs(row["observation"] - month),
                sha256(f"{geo}:{month}:{row['series']}:{row['observation']}:{row['vintage']}".encode()).digest())
    return sorted(common, key=score), sorted(foreign, key=score)


def fixture(source, indicator, selected_months=None, days=CUTOFFS):
    code, geos, dimensions, fixed = INDICATORS[indicator]
    release, values = records(source, geos, dimensions, fixed)
    groups = defaultdict(list)
    for series, month, published, value in values:
        groups[series, month].append({"series": series, "observation": month, "vintage": published,
                                       "release_day": date.fromordinal(published).isoformat(), "value": value})
    for group in groups.values():
        group.sort(key=lambda r: r["vintage"])
    independent = sqlite3.connect(":memory:")
    independent.execute("CREATE TABLE vintage(geo TEXT, observed INT, published INT, value TEXT, "
                        "PRIMARY KEY(geo,observed,published))")
    independent.executemany("INSERT INTO vintage VALUES(?,?,?,?)", values)
    rows = []
    counts = Counter()
    availability = Counter()
    changes = Counter()
    months = [month_id(m) for m in selected_months] if selected_months else [YEARS[0] * 12, (YEARS[1] + 1) * 12 - 1]
    assert months[0] <= months[1] and 0 < days[0] < days[1]
    for geo in geos:
        for month in range(months[0], months[1] + 1):
            year, offset = divmod(month, 12)
            observed_end = date(year, offset + 1, monthrange(year, offset + 1)[1])
            cutoffs = [(observed_end + timedelta(days=d)).toordinal() for d in days]
            history = groups[geo, month]
            options = [r["vintage"] for r in history]
            borders = [bisect_right(options, cutoff) for cutoff in cutoffs]
            focus = sorted({i for boundary in borders for i in range(max(0, boundary-2), min(len(history), boundary+2))})
            exact = [history[i-1] if i else None for i in borders]
            if all(exact):
                changes["numerical_change" if exact[0]["value"] != exact[1]["value"] else "unchanged_value"] += 1
            donor_home, donor_abroad = choose_donors(groups, geos, geo, month, cutoffs)
            assert len(donor_home) >= 16 and len(donor_abroad) >= 24
            selected = [history[i] for i in focus]
            # Up to eight same-address snapshots and an equal mix of local and foreign distractors.
            remaining = 16 - len(selected)
            home, abroad = remaining // 2, (remaining + 1) // 2
            clean = selected + donor_home[:home] + donor_abroad[:abroad]
            spare = [r for r in history if r not in selected]
            spare.sort(key=lambda r: (min(abs(r["vintage"] - t) for t in cutoffs), r["vintage"]))
            extras = spare[:8]
            count = 32 - len(extras)
            noise = clean + extras + donor_home[home:home+count//2] + donor_abroad[abroad:abroad+(count+1)//2]
            assert len(noise) == len({(r["series"], r["observation"], r["vintage"]) for r in noise}) == 48
            questions = []
            for label, cutoff, want in zip(("earlier", "later"), cutoffs, exact, strict=True):
                observed = independent.execute(
                    "SELECT published, value FROM vintage WHERE geo=? AND observed=? AND published <= ?"
                    " ORDER BY published DESC LIMIT 1", (geo, month, cutoff)).fetchone()
                assert observed == ((want["vintage"], want["value"]) if want else None)
                availability[f"{label}_{'known' if want else 'missing'}"] += 1
                position = clean.index(want) if want else None
                questions.append({"period": label, "style": "canonical", "cutoff": cutoff, "gold": position,
                                  "value": want["value"] if want else None, "cutoff_day": date.fromordinal(cutoff).isoformat()})
            rows.append({"id": f"{geo}-{month}", "series": geo, "observation": month, "candidates": noise,
                         "questions": questions, "scenarios": {"clean": list(range(16)), "noise": list(range(48))}})
            counts[geo] += 1
    input_sha = sha256(source.read_bytes()).hexdigest()
    meta = {"source_url": API + code, "dataset": code, "source_sha256": input_sha, "version_timestamp": release,
            "published_cells": len(values), "series": list(geos), "months": selected_months or YEARS, "days_after_month_end": days,
            "observation_coordinate": "calendar_month", "publication_coordinate": "calendar_day",
            "question_availability": dict(availability), "pairs": dict(changes), "groups": dict(counts)}
    return {"source": meta, "rows": {"train": [], "test": rows}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--indicator", choices=INDICATORS, default="unemployment")
    p.add_argument("--months", nargs=2, metavar=("FIRST", "LAST"), help="Inclusive range of YYYY-MM observations.")
    p.add_argument("--days", type=int, nargs=2, default=CUTOFFS)
    args = p.parse_args()
    data = fixture(args.source, args.indicator, args.months, args.days)
    args.output.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    meta = data["source"] | {"fixture_sha256": sha256(args.output.read_bytes()).hexdigest()}
    args.manifest.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
