#!/usr/bin/env python3
"""Construct bitemporal contrast and abstention probes from Philadelphia Fed vintages."""

import argparse
from bisect import bisect_right
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from random import Random
import re
import sqlite3
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from audit_kam_natural_data import digest, stable

SERIES = {
    "ROUTPUT": ("real gross domestic product", "real GDP", "inflation-adjusted aggregate output"),
    "NOUTPUT": ("nominal gross domestic product", "nominal GDP", "current-dollar aggregate output"),
    "RCON": ("real personal consumption expenditures", "real PCE", "inflation-adjusted household spending"),
    "NCON": ("nominal personal consumption expenditures", "nominal PCE", "current-dollar household spending"),
}
ORDINAL = ("first", "second", "third", "fourth")
XML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def period(year, q):
    return int(year) * 4 + int(q) - 1


def format_quarter(number, style):
    year, qi = divmod(number, 4)
    if style == "canonical":
        return f"{year} Q{qi + 1}"
    if style == "paraphrase":
        return f"the {ORDINAL[qi]} quarter of {year}"
    raise ValueError(style)


def read_series(path, series):
    with ZipFile(path) as archive:
        strings = ["".join(node.text or "" for node in si.iter(f"{XML}t"))
                   for si in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        matrix = []
        for row in sheet.iter(f"{XML}row"):
            cells = {}
            for sq in row:
                value = sq.findtext(f"{XML}v")
                if value is None:
                    continue
                col = re.match("[A-Z]+", sq.attrib["r"])[0]
                cells[col] = strings[int(value)] if sq.attrib.get("t") == "s" else value
            if cells:
                matrix.append(cells)
    header, *rows = matrix
    assert header["A"] == "DATE"
    vintages = {}
    for col, name in header.items():
        if col == "A":
            continue
        match = re.fullmatch(rf"{series}(\d{{2}})Q([1-4])", name)
        assert match, name
        yy, q = map(int, match.groups())
        vintages[col] = period(1900 + yy if yy >= 65 else 2000 + yy, q)
    assert sorted(set(vintages.values())) == list(vintages.values())
    observations = {}
    for row in rows:
        match = re.fullmatch(r"(\d{4}):Q([1-4])", row["A"])
        assert match, row["A"]
        observation = period(*match.groups())
        values = {}
        for col, recorded in vintages.items():
            try:
                value = Decimal(row.get(col, ""))
            except InvalidOperation:
                continue
            if value.is_finite():
                values[recorded] = format(value.normalize(), "f")
        assert observation not in observations
        observations[observation] = values
    return observations


def key(record):
    return record["series"], record["observation"], record["vintage"]


def record(data, series, obs, vintage):
    return {"series": series, "observation": obs, "vintage": vintage, "value": data[series][obs][vintage]}


def create_sql_index(data):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE cells (series TEXT, observation INTEGER, vintage INTEGER, value TEXT, "
                "PRIMARY KEY (series, observation, vintage)) WITHOUT ROWID")
    con.executemany("INSERT INTO cells VALUES (?,?,?,?)",
                    ((s, o, v, value) for s, observations in data.items()
                     for o, versions in observations.items() for v, value in versions.items()))
    return con


def latest(data, series, obs, cutoff):
    timeline = data[series][obs]
    dates = list(timeline)
    pos = bisect_right(dates, cutoff) - 1
    return record(data, series, obs, dates[pos]) if pos >= 0 else None


def query(series, obs, cutoff, style):
    series_name, short_name, heldout_alias = SERIES[series]
    if style == "canonical":
        text = f"What was archived for {short_name} observed in {format_quarter(obs, style)} as of {format_quarter(cutoff, style)}?"
        return text, short_name
    text = (f"Using what was known by {format_quarter(cutoff, style)}, what figure was listed for "
            f"{heldout_alias} during {format_quarter(obs, style)}?")
    return text, heldout_alias


def fixture_group(data, con, split, series, obs, vintages):
    rng = Random(int(stable(f"kam-fed-vintage-v1:{split}:{series}:{obs}")[:16], 16))
    candidates, count = {}, Counter()

    def add(item, source):
        if item is not None and key(item) not in candidates:
            candidates[key(item)] = {**item, "source": source}
            count[source] += 1

    bounds = {"train": (period(1978, 1), period(2010, 4)),
              "test": (period(2013, 1), period(2025, 4))}[split]
    lower, upper = bounds
    pool_series = tuple(s for s in SERIES if s != "NCON" or split == "test")
    before, after = vintages
    for v in vintages:
        add(record(data, series, obs, v), "revision_pair")
    same = [v for v in data[series][obs] if lower <= v < before]
    for v in same[-5:]:
        add(record(data, series, obs, v), "older_same_cell")
    blueprints = [(s, o) for s in pool_series for o in range(obs - 6, obs + 7) if (s, o) != (series, obs)]
    rng.shuffle(blueprints)
    for s, o in blueprints:
        if o not in data[s]:
            continue
        item = latest(data, s, o, before)
        if item and item["vintage"] >= lower:
            add(item, "neighbor")
        if len(candidates) >= 16:
            break
    if len(candidates) != 16:
        return None
    clean_ids = set(candidates)
    later = [v for v in data[series][obs] if after < v <= upper]
    different = sorted(later, key=lambda v: (data[series][obs][v] == data[series][obs][after], v))
    for v in different[:24]:
        add(record(data, series, obs, v), "future_same_cell")
    for s, o in blueprints:
        if o not in data[s]:
            continue
        for v in list(data[s][o]):
            if after < v <= upper:
                add(record(data, s, o, v), "future_neighbor")
                if len(candidates) == 48:
                    break
        if len(candidates) == 48:
            break
    if len(candidates) != 48:
        return None

    candidate_list = list(candidates.values())
    rng.shuffle(candidate_list)
    scenes = {"clean": [i for i, rec in enumerate(candidate_list) if key(rec) in clean_ids],
              "noise": list(range(len(candidate_list)))}
    cutoff_before_first = min(data[series][obs]) - 1
    questions = []
    for label, cutoff in (("earlier", before), ("later", after), ("unavailable", cutoff_before_first)):
        found = con.execute("SELECT vintage,value FROM cells WHERE series=? AND observation=? AND vintage<=? "
                            "ORDER BY vintage DESC LIMIT 1", (series, obs, cutoff)).fetchone()
        expected = latest(data, series, obs, cutoff)
        assert found == (expected["vintage"], expected["value"]) if expected else found is None
        goal = next((i for i, c in enumerate(candidate_list) if expected and key(c) == key(expected)), None)
        assert (goal is None) == (label == "unavailable")
        if goal is not None:
            assert goal in scenes["clean"]
        styles = ("canonical", "paraphrase") if split == "test" else ("canonical",)
        for style in styles:
            q, alias = query(series, obs, cutoff, style)
            questions.append({"period": label, "cutoff": cutoff, "style": style, "text": q,
                              "name": alias, "gold": goal, "value": expected["value"] if expected else None})
    return {"id": f"{series}-{obs}", "series": series, "observation": obs,
            "candidates": candidate_list, "scenarios": scenes, "noise_counts": dict(count), "questions": questions}


def cases(data, database, split, test_years=(2013, 2023)):
    ranges = {"train": (1978, 2007), "test": test_years}
    bounds = (period(ranges[split][0], 1), period(ranges[split][1], 1))
    maximum = period(2010 if split == "train" else 2025, 4)
    result = []
    for series in SERIES:
        if series == "NCON" and split == "train":
            continue
        timeline = data[series]
        for obs, observations in timeline.items():
            if not bounds[0] <= obs < bounds[1]:
                continue
            selected = [(a, b) for a, b in zip(observations, list(observations)[1:])
                        if obs < a < b <= min(obs + 15, maximum - 2) and observations[a] != observations[b]]
            options = sorted(selected, key=lambda p: stable(f"kam-fed-pair-v1:{series}:{obs}:{p[0]}"))
            if options and (episode := fixture_group(data, database, split, series, obs, options[0])):
                result.append(episode)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catalog-output", type=Path)
    p.add_argument("--test-years", type=int, nargs=2, metavar=("FIRST", "EXCLUSIVE_LAST"))
    a = p.parse_args()
    if a.test_years is not None and not (2013 <= a.test_years[0] < a.test_years[1] <= 2026):
        p.error("A custom test window must start in or after 2013 and end by 2026")
    sources = {s: a.source_directory / f"fed-{s}.xlsx" for s in SERIES}
    data = {s: read_series(path, s) for s, path in sources.items()}
    database = create_sql_index(data)
    years = tuple(a.test_years) if a.test_years else (2013, 2023)
    sets = {part: cases(data, database, part, years) for part in ("train", "test")}
    assert len(sets["train"]) >= 200 and len(sets["test"]) >= (16 if a.test_years else 100)
    assert {r["id"] for r in sets["train"]}.isdisjoint(r["id"] for r in sets["test"])
    output = {"source_sha256": {s: digest(p) for s, p in sources.items()},
              "archive_nonmissing_cells": {s: sum(map(len, data[s].values())) for s in SERIES},
              "counts": {part: dict(Counter(x["series"] for x in rows)) for part, rows in sets.items()},
              "rows": sets}
    if a.test_years:
        output["test_years"] = list(years)
    a.output.write_text(json.dumps(output, separators=(",", ":"), ensure_ascii=False) + "\n")
    if a.catalog_output:
        archive = [record(data, s, o, v) for s, observations in data.items()
                   for o, versions in observations.items() for v in versions]
        a.catalog_output.write_text(json.dumps({"source_sha256": output["source_sha256"], "rows": archive},
                                              separators=(",", ":")) + "\n")
    print({k: v for k, v in output.items() if k != "rows"})


if __name__ == "__main__":
    main()
