#!/usr/bin/env python3
"""Audit an as-ingested index against scan and SQLite under delayed arrivals."""

import argparse
from bisect import bisect_right, insort
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import sqlite3
from statistics import median
from time import perf_counter_ns

from audit_kam_natural_data import digest


class HistoryTree:
    def __init__(self, rows):
        ordered = sorted(rows)
        self.vintages = [r[0] for r in ordered]
        self.payloads = [r[2] for r in ordered]
        size = 1 << (len(ordered) - 1).bit_length()
        self.size = size
        self.minimum = [float("inf")] * (2 * size)
        self.minimum[size:size+len(ordered)] = [r[1] for r in ordered]
        for i in range(size - 1, 0, -1):
            self.minimum[i] = min(self.minimum[i * 2], self.minimum[i * 2 + 1])

    def lookup(self, vintage, known):
        cap = bisect_right(self.vintages, vintage)
        if cap == 0:
            return None
        node = self.size + cap - 1
        if self.minimum[node] <= known:
            return self.payloads[cap - 1]
        while node > 1:
            if node & 1 and self.minimum[node - 1] <= known:
                node -= 1
                while node < self.size:
                    tail = node * 2 + 1
                    node = tail if self.minimum[tail] <= known else tail - 1
                return self.payloads[node - self.size]
            node //= 2
        return None


def query_panel(index, seed, n, style):
    rng = Random(seed)
    keys = sorted(index)
    output = []
    for _ in range(n):
        key = rng.choice(keys)
        history = index[key]
        vintage, arrival, _ = rng.choice(history)
        if style == "point":
            seen = vintage * 1000 + rng.randrange(1000)
            valid_cutoff = vintage
        elif style == "independent":
            seen = arrival + rng.randint(-2000, 2000)
            valid_cutoff = vintage + rng.randint(-5, 3)
        else:
            raise ValueError(style)
        output.append((key, valid_cutoff, seen))
    return output


def timed_query(method, queries):
    outputs, timing = [], []
    for _ in range(3):
        start = perf_counter_ns()
        results = [method(*q) for q in queries]
        timing.append((perf_counter_ns() - start) / len(queries))
        if outputs:
            assert results == outputs
        outputs = results
    return outputs, {"ns_per_query": median(timing), "timed_repeats": timing}


def timeline_methods(rows, queries, index, tree):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE cells (address INTEGER, vintage INTEGER, arrived INTEGER, id INTEGER, "
                "PRIMARY KEY (address, vintage)) WITHOUT ROWID")
    con.executemany("INSERT INTO cells VALUES (?,?,?,?)", rows)
    con.commit()

    def scan(address, vintage, known):
        return next((i for v, arrived, i in reversed(index[address]) if v <= vintage and arrived <= known), None)

    def query_sql(address, vintage, known):
        result = con.execute("SELECT id FROM cells WHERE address=? AND vintage<=? AND arrived<=? "
                             "ORDER BY vintage DESC LIMIT 1", (address, vintage, known)).fetchone()
        return result[0] if result else None

    def ignore_ingestion(address, vintage, _):
        bank = tree[address]
        i = bisect_right(bank.vintages, vintage) - 1
        return bank.payloads[i] if i >= 0 else None

    methods = {
        "descending_scan": scan,
        "bisect_static_vintage_only": ignore_ingestion,
        "min_arrival_tree": lambda key, vintage, known: tree[key].lookup(vintage, known),
        "sqlite_index": query_sql,
    }
    result = {}
    outputs = {}
    for name, function in methods.items():
        outputs[name], result[name] = timed_query(function, queries)
    gold = outputs["descending_scan"]
    assert outputs["min_arrival_tree"] == gold and outputs["sqlite_index"] == gold
    for name, guesses in outputs.items():
        result[name].update(correct=sum(a == b for a, b in zip(guesses, gold, strict=True)),
                            unavailable=sum(g is None for g in guesses))
    con.close()
    return gold, result


def stream(rows, queries, groundtruth):
    backlog = sorted(rows, key=lambda r: (r[2], r[0], r[1]))
    requests = sorted(enumerate(queries), key=lambda q: q[1][2])
    active = defaultdict(list)
    outputs = [None] * len(requests)
    pos = 0
    ingest_ns, read_ns = 0, 0
    max_arrival = max(rows, key=lambda r: r[2])[2]
    out_of_order = 0
    before = perf_counter_ns()
    for i, (key, vintage, seen) in requests:
        while pos < len(backlog) and backlog[pos][2] <= seen:
            address, published, _, payload = backlog[pos]
            start = perf_counter_ns()
            line = active[address]
            if line and line[-1][0] > published:
                out_of_order += 1
            insort(line, (published, payload))
            ingest_ns += perf_counter_ns() - start
            pos += 1
        start = perf_counter_ns()
        line = active.get(key, ())
        latest = bisect_right(line, (vintage, float("inf"))) - 1
        outputs[i] = line[latest][1] if latest >= 0 else None
        read_ns += perf_counter_ns() - start
    interleaved_ns = perf_counter_ns() - before
    assert outputs == groundtruth
    return {"queries": len(requests), "ingested": pos, "total_events": len(backlog),
            "unreleased_after_last_request": len(backlog) - pos, "max_event_arrival": max_arrival,
            "out_of_order_insertions": out_of_order, "live_read_ns_per_query": read_ns / len(requests),
            "incremental_ingest_ns_per_arrival": ingest_ns / pos, "interleaved_ns": interleaved_ns}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--queries", type=int, default=40_000)
    a = p.parse_args()
    data = json.loads(a.catalog.read_text())
    addresses, rows, by_address = {}, [], defaultdict(list)
    rng = Random(a.seed)
    for id_, cell in enumerate(data["rows"]):
        address = addresses.setdefault((cell["series"], cell["observation"]), len(addresses))
        vintage = cell["vintage"]
        noise = rng.random()
        delay = rng.randrange(2500, 12500) if noise < 0.04 else rng.randrange(0, 550)
        seen = vintage * 1000 + delay
        rows.append((address, vintage, seen, id_))
        by_address[address].append((vintage, seen, id_))
    assert all(r == sorted(r) and len({v for v, _, _ in r}) == len(r) for r in by_address.values())
    start = perf_counter_ns()
    trees = {k: HistoryTree(r) for k, r in by_address.items()}
    build_ns = perf_counter_ns() - start
    cases = {}
    for i, name in enumerate(("point", "independent")):
        queries = query_panel(by_address, a.seed + 15_871 + i, a.queries, name)
        truth, modes = timeline_methods(rows, queries, by_address, trees)
        cases[name] = {"count": len(queries),
                       "query_sha256": sha256(json.dumps(queries, separators=(",", ":")).encode()).hexdigest(),
                       "gold_sha256": sha256(json.dumps(truth, separators=(",", ":")).encode()).hexdigest(),
                       "methods": modes, "prequential": stream(rows, queries, truth)}
    results = {"seed": a.seed, "records": len(rows), "addresses": len(addresses), "catalog_sha256": digest(a.catalog),
               "date_tick_granularity": "quarter split into 1000 synthetic ingestion ticks",
               "late_arrivals": sum(t - p * 1000 >= 2500 for _, p, t, _ in rows),
               "static_tree_build_ns": build_ns, "profiles": cases}
    a.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
