"""Timed FIFO arrival source for the in-process serving experiment."""

from collections import deque
from random import Random
from threading import Condition, Thread
from time import perf_counter, sleep

SIZES = (1, 2, 4, 8)
REQUESTS = 384
SEEDS = (17590, 27055)
POLICIES = ((15, 1, 0), (30, 8, 5), (30, 1, 0), (15, 8, 5))


def arrivals(rate, seed, cases):
    timing, selection = Random(seed), Random(seed + 30119)
    elapsed = 0.0
    for _ in range(REQUESTS):
        elapsed += timing.expovariate(rate)
        yield elapsed, selection.choice(cases)


def pack_decision(row):
    return {"selected_ids": [row["ranked"][i] for i in row["selected"]], "prediction": row["prediction"]}


def exercise(request, cases, rate, cap, wait_ms, seed, *, widths=SIZES, pack=pack_decision):
    trace = list(arrivals(rate, seed, cases))
    start = perf_counter() + 0.05
    pending = deque()
    condition = Condition()
    done = False

    def produce():
        nonlocal done
        for seq, (when, question) in enumerate(trace):
            delay = start + when - perf_counter()
            if delay > 0:
                sleep(delay)
            with condition:
                pending.append((seq, when, perf_counter() - start, question))
                condition.notify()
        with condition:
            done = True
            condition.notify()

    source = Thread(target=produce, daemon=True)
    source.start()
    batches = []
    handled = 0
    while handled < len(trace):
        with condition:
            condition.wait_for(lambda: pending)
            expiry = start + pending[0][2] + wait_ms / 1000
            condition.wait_for(lambda: len(pending) >= cap or done, timeout=max(0, expiry - perf_counter()))
            size = max(n for n in widths if n <= min(len(pending), cap))
            batch = [pending.popleft() for _ in range(size)]
            begin = perf_counter() - start
        cell, rows = request([q for _, _, _, q in batch])
        finished = perf_counter() - start
        records = []
        for (seq, target, observed, _), row in zip(batch, rows, strict=True):
            records.append({"seq": seq, "id": row["id"], "target": target, "arrival": observed, **pack(row)})
        batches.append({"begin": begin, "finished": finished, "service": cell["seconds"], "rows": records})
        handled += size
    source.join()
    return {"rate": rate, "max_batch": cap, "max_wait_ms": wait_ms, "seed": seed, "batches": batches}
