#!/usr/bin/env python3
"""Seal authored as-of requests over two official vintage sources with overlapping countries."""

import argparse
from calendar import month_name, monthrange
from collections import Counter
from datetime import date, timedelta
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3

from prepare_kam_eurostat_vintages import API, INDICATORS, records

SOURCE_SHA = {"unemployment": "43ce9212b1a16edd4eb5a4e20b23686f92107553aea5841e3ac184b26fbc7d0f",
              "industry": "f39e475937afb8eda7410058cf774312899a1770ac04cecc8ea1ac8b78ccfac9"}
COUNTRIES = {"HU": "Hungary", "IE": "Ireland", "PT": "Portugal", "RO": "Romania", "SI": "Slovenia", "SK": "Slovakia"}
DEV = ("HU", "IE", "PT")
TEST = ("RO", "SI", "SK")
NAMES = {"unemployment": "seasonally adjusted unemployment rate",
         "industry": "seasonally adjusted month-on-month industrial production growth"}
SEMANTIC = {
    "development": {"unemployment": ("percentage of the active population without work and looking for a job", "jobless rate"),
                    "industry": ("change in total factory output compared with the previous month", "industrial output growth")},
    "test": {"unemployment": ("share of the labor force recorded as unemployed", "joblessness percentage"),
             "industry": ("change in production across industry excluding construction from the prior month", "monthly factory activity growth")},
}
UNKNOWN_METRICS = {
    "development": ("consumer price inflation", "unemployment benefit expenditure", "long-term government interest rates"),
    "test": ("retail sales volume growth", "public employment growth", "hourly labor cost inflation"),
}
UNKNOWN_GEOGRAPHIES = {
    "development": ("Germany", "France", "Austria"),
    "test": ("Finland", "Greece", "Lithuania"),
}
MONTH_PATTERN = r"\b(" + "|".join(month_name[1:]) + r")\s+(20\d\d)\b"
ISO_DAY = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
ISO_MONTH = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])(?![-\d])\b")
OFFSET = re.compile(r"\b(90|365) days after that month ended\b")


def month_number(year, month):
    return year * 12 + month - 1


def month_text(index):
    year, part = divmod(index, 12)
    return f"{month_name[part+1]} {year}"


def parse_dates(text):
    days = ISO_DAY.findall(text)
    offsets = OFFSET.findall(text)
    words = re.findall(MONTH_PATTERN, text)
    numeric = ISO_MONTH.findall(text)
    assert len(words) + len(numeric) == len(days) + len(offsets) == 1
    if words:
        name, year_ = words[0]
        year, month = int(year_), list(month_name).index(name)
    else:
        year, month = map(int, numeric[0])
    if days:
        cutoff = date.fromisoformat(days[0])
    else:
        cutoff = date(year, month, monthrange(year, month)[1]) + timedelta(days=int(offsets[0]))
    return month_number(year, month), cutoff.toordinal()


def remove_dates(text):
    text = ISO_DAY.sub(" ", text)
    text = ISO_MONTH.sub(" ", text)
    text = OFFSET.sub(" ", text)
    return re.sub(MONTH_PATTERN, " ", text)


def registry_country(text):
    found = {key for key, name in COUNTRIES.items()
             if re.search(rf"\b{name}\b", text, re.I) or re.search(rf"\b{key}\b", text)}
    return next(iter(found)) if len(found) == 1 else None


def registry_indicator(text):
    haystack = text.casefold()
    found = {k for k, v in NAMES.items() if v in haystack}
    return next(iter(found)) if len(found) == 1 else None


def render(country, indicator, observation, cutoff, offset, style, split):
    name = COUNTRIES[country]
    month = month_text(observation)
    iso_month = f"{observation//12:04d}-{observation%12+1:02d}"
    iso_date = date.fromordinal(cutoff).isoformat()
    literal = NAMES[indicator]
    para, short = SEMANTIC[split][indicator]
    variants = {
        "canonical": f"What was the {literal} for {name} in {month}, using releases available through {iso_date}?",
        "relative": f"For {name}, what was the {literal} in {month} as recorded {offset} days after that month ended?",
        "paraphrase": f"By {iso_date}, what was the {para} in {name} for {month}?",
        "country_code": f"For {country}, in {iso_month}, report the {short} using releases through {iso_date}.",
    }
    return variants[style]


def document(indicator, country, observation, release, value):
    absolute = date.fromordinal(release).isoformat()
    return (f"{COUNTRIES[country]} ({country}), {NAMES[indicator]}. "
            f"Reference month {month_text(observation)}. Published {absolute}. Measured value {value} percent.")


def card(indicator, country):
    name = COUNTRIES[country]
    return f"{name} ({country}). {NAMES[indicator]}. Monthly percentage, European statistical vintages."


def build(a):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE vintages(id INTEGER, indicator TEXT, geo TEXT, observation INT, publication INT, value TEXT, "
               "PRIMARY KEY(indicator,geo,observation,publication))")
    corpus, source = [], {}
    for indicator, (code, _, dims, fixed) in INDICATORS.items():
        file = getattr(a, indicator)
        stamp = sha256(file.read_bytes()).hexdigest()
        assert stamp == SOURCE_SHA[indicator]
        updated, cells = records(file, tuple(COUNTRIES), dims, fixed)
        source[indicator] = {"source_sha256": stamp, "dataset": code, "cells": len(cells),
                             "updated": updated, "url": API + code, "fixed_dimensions": fixed}
        for geo, month, day, value in sorted(cells):
            i = len(corpus)
            db.execute("INSERT INTO vintages VALUES(?,?,?,?,?,?)", (i, indicator, geo, month, day, value))
            corpus.append({"indicator": indicator, "country": geo, "observation": month,
                           "publication": day, "value": value})

    def target(indicator, geo, obs, day):
        got = db.execute("SELECT id FROM vintages WHERE indicator=? AND geo=? AND observation=? AND publication<=? "
                         "ORDER BY publication DESC LIMIT 1", (indicator, geo, obs, day)).fetchone()
        return None if got is None else got[0]

    q = {}
    for split, countries in (("development", DEV), ("test", TEST)):
        months = (range(month_number(2023, 1), month_number(2023, 7)) if split == "development"
                  else range(month_number(2024, 1), month_number(2025, 7)))
        rows = []

        def add(text, style, kind, month, cutoff, group, gold=None):
            parsed = parse_dates(text)
            assert parsed == (month, cutoff)
            sha = sha256(text.encode()).hexdigest()
            rows.append({"id": sha, "text": text, "style": style, "kind": kind, "group": group,
                         "observation": month, "cutoff": cutoff, "gold": gold})

        for country in countries:
            for obs in months:
                month = month_text(obs)
                y, m = divmod(obs, 12)
                end = date(y, m+1, monthrange(y, m+1)[1])
                for offset in (90, 365):
                    cutoff = (end + timedelta(days=offset)).toordinal()
                    present = {k: target(k, country, obs, cutoff) for k in INDICATORS}
                    assert all(row is not None for row in present.values())
                    for k, j in present.items():
                        group = f"{country}/{obs}/{k}"
                        for style in ("canonical", "relative", "paraphrase", "country_code"):
                            text = render(country, k, obs, cutoff, offset, style, split)
                            assert registry_country(text) == country
                            add(text, style, "answerable", obs, cutoff, group, j)
                    iso = date.fromordinal(cutoff).isoformat()
                    c = COUNTRIES[country]
                    raw = (f"For {c} in {month}, what was the seasonally adjusted monthly percentage "
                           f"reported in releases through {iso}?")
                    add(raw, "missing_indicator", "ambiguous", obs, cutoff, f"{country}/{obs}/ambiguous")
                    idx = (obs + offset + list(countries).index(country)) % len(UNKNOWN_METRICS[split])
                    other = UNKNOWN_METRICS[split][idx]
                    raw = f"For {c} in {month}, what was the {other} based on releases through {iso}?"
                    add(raw, "unsupported_indicator", "unknown", obs, cutoff, f"{country}/{obs}/unsupported_indicator")
                    geography = UNKNOWN_GEOGRAPHIES[split][list(countries).index(country)]
                    raw = f"What was the {NAMES['unemployment']} for {geography} in {month}, using releases available through {iso}?"
                    add(raw, "unsupported_country", "unknown", obs, cutoff, f"{geography}/{obs}/unsupported_country")
        assert len(rows) == len({r["id"] for r in rows})
        q[split] = rows
    assert {x["id"] for x in q["development"]}.isdisjoint(r["id"] for r in q["test"])
    for rows in q.values():
        for item in rows:
            if item["gold"] is not None:
                row = corpus[item["gold"]]
                assert row["observation"] == item["observation"] and row["publication"] <= item["cutoff"]
    return q, corpus, source


def bundle_hash(rows):
    return sha256(json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("unemployment", "industry", "output", "manifest"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    questions, corpus, source = build(a)
    records = {"sources": source, "corpus": corpus, "questions": questions}
    a.output.write_bytes(gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0))
    manifest = {"sources": source, "countries": COUNTRIES, "countries_development": DEV, "countries_test": TEST,
                "documents": len(corpus), "corpus_content_sha256": bundle_hash(corpus),
                "fixture_sha256": sha256(a.output.read_bytes()).hexdigest(), "dates": {"days_after_month_end": [90, 365]},
                "questions": {k: dict(Counter(r["style"] for r in v)) for k, v in questions.items()}}
    a.manifest.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
