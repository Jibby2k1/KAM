#!/usr/bin/env python3
"""Separate lexical scoring from parsing and test a frozen term-pruning arm."""

import argparse
import json
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import tantivy

from audit_kam_natural_data import digest, words

ARMS = ("parser", "direct", "df_35", "df_20", "df_12")
PRUNING = {"df_35": .35, "df_20": .20, "df_12": .12}


def lexical_terms(question, searcher, max_df=None):
    terms = sorted(set(words(question)))
    if max_df is None:
        return terms
    ceiling = max_df * searcher.num_docs
    kept = [term for term in terms if searcher.doc_freq("contents", term) <= ceiling]
    return kept if len(kept) >= 3 else terms


def parse_terms(index, terms):
    return index.parse_query(" ".join(f'"{term}"' for term in terms), ["contents"])


def cutoff_hits(searcher, query, depth):
    limit = depth + 1
    while True:
        hits = searcher.search(query, limit=limit, count=False).hits
        if len(hits) < limit or hits[-1][0] < hits[depth - 1][0]:
            return hits
        limit *= 2
        assert limit <= 8192


def stable_ids(searcher, hits, depth):
    ids = searcher.fast_field_values("row_id", [address for _, address in hits])
    return sorted(((i, hit[0]) for i, hit in zip(ids, hits, strict=True)), key=lambda pair: (-pair[1], pair[0]))[:depth]


def run(a):
    qs, fixture, metadata = (json.loads(path.read_text()) for path in (a.queries, a.fixture, a.metadata))
    assert fixture["previous_queries_sha256"] == metadata["queries_sha256"] == digest(a.queries)
    assert fixture["source_sha256"] == metadata["source_sha256"]
    assert metadata["fast_row_id"]
    groups = {"calibration": qs["calibration"], "test": fixture["rows"]["test"]}
    index = tantivy.Index.open(str(a.index))
    searcher = index.searcher()
    assert searcher.num_docs == metadata["indexed"]
    depth = 128
    rng = Random(182900)

    def request(q, arm):
        start = perf_counter()
        terms = lexical_terms(q["question"], searcher, PRUNING.get(arm))
        safe_direct = all(term.isascii() and term.isalnum() and len(term) <= 40 for term in terms)
        if arm == "direct" and safe_direct and terms:
            clauses = [tantivy.Query.term_query(index.schema, "contents", t, index_option="freq") for t in terms]
            query = clauses[0] if len(clauses) == 1 else tantivy.Query.boolean_query(
                [(tantivy.Occur.Should, clause) for clause in clauses])
        else:
            query = parse_terms(index, terms)
        parsed = perf_counter()
        hits = cutoff_hits(searcher, query, depth)
        ranked = perf_counter()
        result = stable_ids(searcher, hits, depth)
        end = perf_counter()
        return {"id": q.get("id", q.get("_id")), "arm": arm, "terms": len(terms),
                "direct_used": arm == "direct" and safe_direct and bool(terms), "ranked": result,
                "parse": parsed - start, "rank": ranked - parsed, "fetch": end - ranked, "seconds": end - start}

    rows = {}
    for arm in ARMS:
        request(groups["calibration"][0], arm)
    # Calibration questions have no row IDs. Resolve only their pinned gold titles.
    gold = {}
    for q in groups["calibration"] + groups["test"]:
        id_ = q.get("id", q.get("_id"))
        gold[id_] = q["gold_ids"] if "gold_ids" in q else None
    cal_ids = json.loads(a.calibration_gold.read_text())
    assert {k for k, v in gold.items() if v is None} == cal_ids.keys()
    gold.update(cal_ids)
    for label, group in groups.items():
        use = ARMS if label == "calibration" else ("parser", "direct", chosen)
        rows[label] = []
        for q in group:
            for arm in rng.sample(list(dict.fromkeys(use)), len(set(use))):
                record = request(q, arm)
                record["complete"] = set(gold[record["id"]]) <= {i for i, _ in record["ranked"]}
                rows[label].append(record)
        if label == "calibration":
            original = sum(r["complete"] for r in rows[label] if r["arm"] == "parser")
            eligible = [arm for arm in PRUNING if sum(r["complete"] for r in rows[label] if r["arm"] == arm) >= original - 1]
            chosen = min(eligible, key=lambda arm: median(r["seconds"] for r in rows[label] if r["arm"] == arm)) if eligible else "parser"
    output = {"status": "completed", "chosen": chosen, "depth": depth, "queries_sha256": digest(a.queries),
              "fixture_sha256": digest(a.fixture), "calibration_gold_sha256": digest(a.calibration_gold),
              "metadata_sha256": digest(a.metadata), "rows": rows}
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "fixture", "calibration-gold", "metadata", "index", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
