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

DATASET = "ei_lm_m_vtg"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/" + DATASET
GEOS = ("DE", "ES", "FR", "IT", "NL", "PL")
DIMENSIONS = ("freq", "unit", "s_adj", "sex", "age", "geo", "revdate", "time")
KEY = ("M", "PC_ACT", "SA", "T", "TOTAL")
YEARS = (2021, 2024)
CUTOFFS = (90, 365)


def month_id(month):
    assert len(month) == 7
    y, m = map(int, month.split("-"))
    assert 1 <= m <= 12
    return y * 12 + m - 1


def records(source):
    payload = json.loads(source.read_bytes())
    assert tuple(payload["id"]) == DIMENSIONS
    shapes = payload["size"]
    categories = []
    for key, size in zip(payload["id"], shapes, strict=True):
        indexed = payload["dimension"][key]["category"]["index"]
        assert set(indexed.values()) == set(range(size))
        categories.append(sorted(indexed, key=indexed.get))
    assert [v[0] for v in categories[:5]] == list(KEY) and shapes[:5] == [1] * 5
    assert set(categories[5]) == set(GEOS)
    results = []
    for loc, value in payload["value"].items():
        assert type(value) in (float, int)
        ix = int(loc)
        indices = []
        for dim in reversed(shapes):
            ix, rem = divmod(ix, dim)
            indices.append(rem)
        assert ix == 0
        _, _, _, _, _, country, day, month = [labels[j] for labels, j in
                                               zip(categories, reversed(indices), strict=True)]
        results.append((country, month_id(month), date.fromisoformat(day).toordinal(), str(value)))
    return payload["updated"], results


def choose_donors(by_month, geo, month, cutoffs):
    common, foreign = [], []
    for neighbor in range(month - 5, month + 6):
        for side in GEOS:
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


def fixture(source):
    release, values = records(source)
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
    for geo in GEOS:
        for month in range(YEARS[0] * 12, (YEARS[1] + 1) * 12):
            year, offset = divmod(month, 12)
            observed_end = date(year, offset + 1, monthrange(year, offset + 1)[1])
            cutoffs = [(observed_end + timedelta(days=d)).toordinal() for d in CUTOFFS]
            history = groups[geo, month]
            options = [r["vintage"] for r in history]
            borders = [bisect_right(options, cutoff) for cutoff in cutoffs]
            focus = sorted({i for boundary in borders for i in range(max(0, boundary-2), min(len(history), boundary+2))})
            exact = [history[i-1] if i else None for i in borders]
            if all(exact):
                changes["numerical_change" if exact[0]["value"] != exact[1]["value"] else "unchanged_value"] += 1
            donor_home, donor_abroad = choose_donors(groups, geo, month, cutoffs)
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
    meta = {"source_url": API, "dataset": DATASET, "source_sha256": input_sha, "version_timestamp": release,
            "published_cells": len(values), "series": list(GEOS), "months": YEARS, "days_after_month_end": CUTOFFS,
            "observation_coordinate": "calendar_month", "publication_coordinate": "calendar_day",
            "question_availability": dict(availability), "pairs": dict(changes), "groups": dict(counts)}
    return {"source": meta, "rows": {"train": [], "test": rows}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    data = fixture(args.source)
    args.output.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    meta = data["source"] | {"fixture_sha256": sha256(args.output.read_bytes()).hexdigest()}
    args.manifest.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
