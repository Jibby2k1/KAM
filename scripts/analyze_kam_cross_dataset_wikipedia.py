#!/usr/bin/env python3
"""Audit cross-dataset title, annotated-span and excerpt coverage."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from analyze_kam_shared_serving import unpack


def evaluate(rows):
    totals = defaultdict(int)
    for row in rows:
        units = row["units"]
        titles = tuple({unit["title_sha256"]: unit for unit in units}.values())
        assert units and all(unit["words"] > 0 for unit in units)
        assert all(not (t["indexed"] and t["excluded"]) for t in titles)
        assert all(unit.get("shared_excerpt", False) >= unit.get("exact_unit", False) for unit in units)
        assert all(unit.get("body_words", 0) >= unit["words"] for unit in units if unit.get("exact_unit"))
        totals["questions"] += 1
        totals["all_titles"] += all(t["indexed"] for t in titles)
        totals["any_permanent_exclusion"] += any(t["excluded"] for t in titles)
        totals["any_absent_without_permanent_exclusion"] += any(not (t["excluded"] or t["indexed"]) for t in titles)
        totals["all_exact_units"] += all(u.get("exact_unit", False) for u in units)
        totals["all_units_have_excerpt"] += all(u.get("shared_excerpt", False) for u in units)
        totals["any_exact_unit"] += any(u.get("exact_unit", False) for u in units)
        totals["annotated_units"] += len(units)
        totals["indexed_units"] += sum(u["indexed"] for u in units)
        totals["exact_units"] += sum(u.get("exact_unit", False) for u in units)
        totals["units_with_excerpt"] += sum(u.get("shared_excerpt", False) for u in units)
    assert 0 <= totals["all_exact_units"] <= totals["all_units_have_excerpt"] <= totals["all_titles"]
    return dict(totals)


def summarize(raw):
    assert set(raw["rows"]) == set(raw["data_sha256"]) == {"twowiki", "musique"}
    summary = {}
    for dataset, cases in raw["rows"].items():
        assert len(cases) == len({r["id"] for r in cases})
        kinds, title_counts = defaultdict(list), defaultdict(list)
        for row in cases:
            kinds[row["type"]].append(row)
            title_counts[len({u["title_sha256"] for u in row["units"]})].append(row)
        summary[dataset] = {"all": evaluate(cases), "by_type": {k: evaluate(v) for k, v in kinds.items()},
                            "by_supporting_titles": {str(k): evaluate(v) for k, v in title_counts.items()},
                            "source_records_excluded": raw["excluded_source_records"][dataset]}
    return {"corpus": raw["corpus"], "data_sha256": raw["data_sha256"], "coverage": summary}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coverage", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = summarize(unpack(a.coverage))
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
