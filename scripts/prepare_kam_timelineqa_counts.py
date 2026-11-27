#!/usr/bin/env python3
"""Freeze exact monthly counts and date-only distractor edits in generated lifelogs."""

import argparse
import calendar
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
from random import Random
import sqlite3

from audit_kam_natural_data import digest, stable
from prepare_kam_timelineqa import DATE

SCHEMA = "timelineqa_monthly_counts_v1"
SIZES = {"exercise": ("daily_exercise-log.csv", "exercise", "q8"),
         "lunch": ("daily_meal-log.csv", "foodtype", "q11")}
DEPTH = 128
DONORS = 8


def load(seed, density, path, templates):
    episodes = path / "episodes.json"
    raw = json.loads(episodes.read_text())
    by_id = {v["eid"]: (day, k, v) for day, events in raw.items() for k, v in events.items()}
    assert len(by_id) == sum(map(len, raw.values()))
    cases, hashes, kept = [], {"episodes": digest(episodes)}, {}
    for kind, (csv_name, field, template) in SIZES.items():
        path_to_rows = path / csv_name
        hashes[csv_name] = digest(path_to_rows)
        with path_to_rows.open(newline="") as json_rows:
            events = [r for r in csv.DictReader(json_rows) if kind != "lunch" or r["mealtype"] == "lunch"]
        rows = {}
        for record in events:
            id_, day = record["eid"], record["date"]
            archived_date, label, note = by_id[id_]
            field_value = record[field]
            assert day == archived_date and DATE.fullmatch(day) and label.startswith(kind)
            assert note["logical_representation"][0] == day and field_value in note["logical_representation"]
            assert id_ not in rows
            rows[id_] = {"id": id_, "day": day, "value": field_value, "text": note["text_template_based"], "kind": kind}
        with sqlite3.connect(":memory:") as db:
            db.execute("create table logs(id text primary key, day text, value text)")
            db.executemany("insert into logs values (?, ?, ?)", ((r["id"], r["day"], r["value"]) for r in rows.values()))
            months = sorted({r["day"][:7] for r in rows.values()})
            values = sorted({r["value"] for r in rows.values()})
            counts = Counter((r["day"][:7], r["value"]) for r in rows.values())
            by_value = defaultdict(list)
            for r in rows.values():
                by_value[r["value"]].append(r["id"])

            def eligible(m, v, low, high):
                n = counts[m, v]
                return low <= n <= high and len(by_value[v]) - n >= DONORS

            ordered = sorted(((m, v) for m in months for v in values),
                             key=lambda pair: stable(f"kam-timeline-count:{seed}:{kind}:{pair}"))
            seen_months = set()
            rng = Random(seed + (917 if kind == "exercise" else 471))
            for low, high, needed in ((2, 12, 6), (0, 0, 2)):
                chosen = 0
                for month, value in ordered:
                    if month in seen_months or not eligible(month, value, low, high):
                        continue
                    year, m = map(int, month.split("/"))
                    ndays = calendar.monthrange(year, m)[1]
                    hits = db.execute("select id from logs where day between ? and ? and value = ? order by id",
                                      (f"{month}/01", f"{month}/{ndays:02}", value)).fetchall()
                    gold = [row[0] for row in hits]
                    label = [id_ for id_, r in rows.items() if r["value"] == value and r["day"].startswith(month + "/")]
                    assert len(gold) == counts[month, value] and set(gold) == set(label)
                    eligible_donors = [i for i in by_value[value] if i not in gold]
                    donors = rng.sample(eligible_donors, DONORS)
                    candidates = set(gold) | set(donors)
                    same_month = [id_ for id_, r in rows.items() if r["day"].startswith(month + "/") and id_ not in candidates]
                    candidates.update(same_month)
                    depth = min(DEPTH, len(rows))
                    assert len(candidates) <= depth
                    candidates.update(rng.sample([id_ for id_ in rows if id_ not in candidates], depth - len(candidates)))
                    ordered_ids = sorted(candidates)
                    rng.shuffle(ordered_ids)
                    replaced = {i: f"{month}/{min(int(rows[i]['day'][-2:]), ndays):02}" for i in donors}
                    q = templates[template].format(exercise=value, foodtype=value, mealtype="lunch",
                                                  year=year, month=calendar.month_name[m])
                    id_ = f"{seed}:{kind}:{month}:{value}"
                    cases.append({"id": id_, "kind": kind, "month": month, "value": value, "question": q,
                                  "gold": gold, "candidates": ordered_ids, "changed_dates": replaced})
                    kept.update({i: rows[i] for i in ordered_ids})
                    seen_months.add(month)
                    chosen += 1
                    if chosen == needed:
                        break
                if low:
                    assert chosen == needed, (seed, density, kind, low, high, chosen)
    return {"seed": seed, "density": density, "files_sha256": hashes, "cases": cases, "episodes": kept}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "personas", "templates", "output", "manifest"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--generator-revision", required=True)
    a = p.parse_args()
    with a.templates.open(newline="") as stream:
        templates = {r[0]: r[4] for r in csv.reader(stream) if r[0] in {"q8", "q11"}}
    packed = [load(int(seed), density, a.source / seed, templates)
              for seed, density in (line.split() for line in a.personas.read_text().splitlines())]
    meta = {"schema": SCHEMA, "generator": a.generator_revision, "template_sha256": digest(a.templates),
            "depth": DEPTH, "changed_dates_per_case": DONORS}
    a.output.write_bytes(gzip.compress(json.dumps(meta | {"personas": packed}, separators=(",", ":"),
                                                  ensure_ascii=False).encode(), mtime=0))
    panel = [{"seed": r["seed"], "density": r["density"], "files_sha256": r["files_sha256"],
              "cases": [{k: q[k] for k in ("id", "kind", "month", "gold", "candidates", "changed_dates")} for q in r["cases"]]}
             for r in packed]
    a.manifest.write_bytes(gzip.compress(json.dumps(meta | {"fixture_sha256": digest(a.output), "personas": panel},
                                                    separators=(",", ":"), ensure_ascii=False).encode(), mtime=0))
    print(Counter((r["density"], q["kind"], bool(q["gold"])) for r in packed for q in r["cases"]))


if __name__ == "__main__":
    main()
