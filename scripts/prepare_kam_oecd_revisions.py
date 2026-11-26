#!/usr/bin/env python3
"""Prepare exact as-of cohorts from the Dallas Fed's GDP vintage matrices."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import xlrd

FILES = {"Canada": "canrealGDP.xls", "France": "frarealGDP.xls", "Italy": "itarealGDP.xls",
         "Portugal": "porrealGDP.xls"}
BASE = "https://www.dallasfed.org/~/media/documents/research/international/oecd/"


def read_matrix(path, country):
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)
    assert sheet.cell_value(0, 0) == country
    dates = [xlrd.xldate_as_datetime(c.value, book.datemode).date() for c in sheet.row(3)[3:]]
    assert dates == sorted(set(dates))
    quarters = [d.year * 4 + (d.month - 1) // 3 for d in dates]
    assert quarters == sorted(set(quarters)), "Each release must have a unique quarter"
    for row in range(4, sheet.nrows):
        q, y = map(int, sheet.row_values(row, 1, 3))
        assert 1 <= q <= 4
        observation = y * 4 + q - 1
        for j, cell in enumerate(sheet.row(row)[3:]):
            if cell.ctype == xlrd.XL_CELL_NUMBER:
                assert quarters[j] > observation
                yield (country, observation, quarters[j], dates[j].isoformat(), repr(cell.value))
            else:
                assert cell.value in ("", ".")


def build(root):
    records, sources = [], {}
    for country, filename in FILES.items():
        p = root / filename
        cells = list(read_matrix(p, country))
        sources[country] = {"url": BASE + filename, "sha256": sha256(p.read_bytes()).hexdigest(),
                            "records": len(cells)}
        records.extend(cells)
    timelines = defaultdict(list)
    for series, obs, vintage, release_day, value in records:
        timelines[series, obs].append({"series": series, "observation": obs, "vintage": vintage,
                                       "release_day": release_day, "value": value})
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE releases(series, observation, vintage, day, val, PRIMARY KEY(series, observation, vintage))")
    con.executemany("INSERT INTO releases VALUES (?,?,?,?,?)", records)
    counters = Counter()
    groups = []
    for (series, obs), history in timelines.items():
        history.sort(key=lambda r: r["vintage"])
        options = [(i, i+1) for i in range(len(history)-1) if history[i]["value"] != history[i+1]["value"]
                   and history[i]["vintage"] - obs <= 16]
        if len(history) < 4 or not options:
            continue
        first, second = min(options, key=lambda x: sha256(f"kam-oecd-v1:{series}:{obs}:{x[0]}".encode()).digest())
        # Recent, same-country observations and other-country rows form the nuisance bank.
        # The ordering is fixed before any model is evaluated.
        donors = [r for (s, o), bank in timelines.items() if (s, o) != (series, obs)
                  and abs(o - obs) <= 8 for r in bank
                  if abs(r["vintage"] - history[first]["vintage"]) <= 16]
        donors.sort(key=lambda r: (abs(r["observation"]-obs) + abs(r["vintage"]-history[first]["vintage"]),
                                   sha256(f"{series}:{obs}:{r['series']}:{r['observation']}:{r['vintage']}".encode()).digest()))
        if len(donors) < 45:
            continue
        core = [history[first], history[second]]
        other = [r for r in history if r not in core]
        other.sort(key=lambda r: (abs(r["vintage"] - history[first]["vintage"]), r["vintage"]))
        # All cutoffs retain their gold in every panel.
        candidates = core + other + donors[:46]
        fill = len(core) + len(other)
        panels = {"observed": (list(range(16)), list(range(48))),
                  "thin": ([0, 1, 2] + list(range(fill, fill+13)), [0, 1, 2] + list(range(fill, fill+45)))}
        questions = []
        for label, i in (("earlier", first), ("later", second), ("unavailable", None)):
            vintage = history[i]["vintage"] if i is not None else history[0]["vintage"] - 1
            pos = bisect_right([r["vintage"] for r in history], vintage) - 1
            expected = history[pos] if pos >= 0 else None
            cutoff_day = history[i]["release_day"] if i is not None else (
                date.fromisoformat(history[0]["release_day"]) - timedelta(days=1)).isoformat()
            found = con.execute("SELECT vintage, val FROM releases WHERE series=? AND observation=? AND day<=? "
                                "ORDER BY day DESC LIMIT 1", (series, obs, cutoff_day)).fetchone()
            assert found == ((expected["vintage"], expected["value"]) if expected else None)
            gold = candidates.index(expected) if expected else None
            questions.append({"period": label, "style": "canonical", "cutoff": vintage,
                              "gold": gold, "value": expected["value"] if expected else None})
        groups.append({"id": f"{series}-{obs}", "series": series, "observation": obs,
                       "candidates": candidates, "panels": panels, "questions": questions})
        counters[series] += 1
    assert all(counters[c] >= 12 for c in FILES), counters
    return {"sources": sources, "test_years": [1956, 1999], "counts": dict(counters), "groups": groups,
            "oracle": {"as_of_known": 2 * len(groups), "as_of_unavailable": len(groups),
                       "method": "independent SQLite lookup at the exact source release day"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-directory", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--training-fixture", required=True, type=Path)
    p.add_argument("--fixture-prefix", required=True, type=Path)
    args = p.parse_args()
    data = build(args.source_directory)
    train = json.loads(args.training_fixture.read_text())["rows"]["train"]
    for mode in ("observed", "thin"):
        panel_rows = []
        for group in data["groups"]:
            assert all(len(set(v)) == n for v, n in zip(group["panels"][mode], (16, 48), strict=True))
            assert all(q["gold"] in idx for q in group["questions"] if q["gold"] is not None
                       for idx in group["panels"][mode])
            small, large = group["panels"][mode]
            panel_rows.append({k: group[k] for k in ("id", "series", "observation", "candidates", "questions")} |
                              {"scenarios": {"clean": small, "noise": large}})
        assert {r["id"] for r in train}.isdisjoint(r["id"] for r in panel_rows)
        fixture = {"test_years": data["test_years"], "sources": data["sources"],
                   "training_sha256": sha256(args.training_fixture.read_bytes()).hexdigest(),
                   "rows": {"train": train if mode == "observed" else [], "test": panel_rows}}
        path = args.fixture_prefix.with_name(args.fixture_prefix.name + f"-{mode}.json")
        path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    args.output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(data["counts"])


if __name__ == "__main__":
    main()
