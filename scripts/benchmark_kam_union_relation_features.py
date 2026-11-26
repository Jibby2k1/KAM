#!/usr/bin/env python3
"""Measure a bitset implementation of relation features on actual HotpotQA unions."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter_ns
import sqlite3

import numpy as np
import torch

from audit_kam_natural_data import digest, words
from run_kam_fullwiki_support_links import STOP, relation_features
from run_kam_sentence_verifier_fullwiki import SENTENCE_BREAK


def bitset_relation_features(example, pairs):
    """Return the existing exact title edges and five lexical relation features."""
    question = set(words(example["question"])) - STOP
    scale = max(len(question), 1)
    bit = {token: 1 << position for position, token in enumerate(question)}
    trie, names, bodies, coverage = {}, [], [], []
    for i, (title, text) in enumerate(example["context"]):
        name = words(title)
        body = words(" ".join(text))[:120]
        names.append(set(name))
        bodies.append(body)
        coverage.append(sum(bit.get(token, 0) for token in (set(body) | names[-1])))
        if len(" ".join(name)) >= 3:
            node = trie
            for token in name:
                node = node.setdefault(token, {})
            node.setdefault(None, []).append(i)
    n = len(bodies)
    graph = np.zeros((n, n), dtype=bool)
    local_counts = np.zeros((n, n), dtype=int)
    for i, body in enumerate(bodies):
        for start, token in enumerate(body):
            node = trie.get(token)
            end = start
            while node is not None:
                for j in node.get(None, ()):
                    graph[i, j] = True
                    window = body[max(0, start - 12):start] + body[end + 1:end + 13]
                    hits = len((set(window) & question) - names[j])
                    local_counts[i, j] = max(local_counts[i, j], hits)
                end += 1
                node = node.get(body[end]) if end < len(body) else None
    left, right = pairs
    sizes = [mask.bit_count() for mask in coverage]
    content = torch.tensor([((coverage[i] | coverage[j]).bit_count() / scale,
                             min(sizes[i], sizes[j]) / scale,
                             (coverage[i] & coverage[j]).bit_count() / scale)
                            for i, j in zip(left.tolist(), right.tolist(), strict=True)])
    local = torch.as_tensor(local_counts / scale, dtype=torch.get_default_dtype())
    fwd, rev = local[left, right], local[right, left]
    return torch.from_numpy(graph), torch.cat((content, torch.stack((torch.maximum(fwd, rev), fwd + rev), -1)), -1)


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def measure(f, example, pairs):
    started = perf_counter_ns()
    graph, features = f(example, pairs)
    return graph, features, (perf_counter_ns() - started) / 1e6


def cpu_model():
    proc = Path("/proc/cpuinfo")
    if proc.exists():
        for line in proc.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                return value.strip()
    return None


def run(a):
    fixture, pools = read(a.fixture), read(a.pools)
    cases = {q["id"]: q for q in fixture["rows"]["test"]}
    assert set(cases) == set(pools["rows"])
    records = sorted((r for s in read(a.replay) for r in s["rows"]), key=lambda r: r["id"])
    assert len(records) == len(cases) == 1024
    # Sample uniformly from the frozen ID order. Both implementations see all variants.
    chosen = records[a.offset::a.stride]
    pairs_by_size, rows = {}, []
    db_uri = f"file:{a.database}?immutable=1"
    with sqlite3.connect(db_uri, uri=True) as db:
        for i, r in enumerate(chosen):
            id_ = r["id"]
            base = pools["rows"][id_]
            expanded = list(dict.fromkeys(base + r["corpus_hits"]["conditioned_union"]))
            slots = ",".join("?" for _ in expanded)
            text = {row: (title, SENTENCE_BREAK.split(body)) for row, title, body in
                    db.execute(f"SELECT id,title,body FROM docs WHERE id IN ({slots})", expanded)}
            assert len(text) == len(expanded)
            for name, candidates in (("original", base), ("conditioned_union", expanded)):
                n = len(candidates)
                if n not in pairs_by_size:
                    pairs_by_size[n] = torch.triu_indices(n, n, 1)
                pairs = pairs_by_size[n]
                example = {"question": cases[id_]["question"], "context": [text[j] for j in candidates]}
                methods = {"baseline": relation_features, "bitset": bitset_relation_features}
                order = tuple(methods) if i % 2 else tuple(reversed(methods))
                outputs = {key: measure(methods[key], example, pairs) for key in order}
                old, new = (outputs[key] for key in methods)
                assert torch.equal(old[0], new[0]) and torch.equal(old[1], new[1]), (id_, name)
                rows.append({"id": id_, "pool": name, "documents": n,
                             "baseline_ms": old[2], "bitset_ms": new[2]})
    summary = {}
    for pool in ("original", "conditioned_union"):
        selected = [r for r in rows if r["pool"] == pool]
        summary[pool] = {key: {"p50_ms": round(median(vals), 6), "p95_ms": round(quantiles(vals, n=20)[18], 6)}
                         for key in ("baseline_ms", "bitset_ms")
                         if (vals := [r[key] for r in selected])}
    return {key + "_sha256": digest(getattr(a, key)) for key in ("fixture", "pools", "replay")} | {
        "cpu_model": cpu_model(), "torch_version": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
        "bitwise_parity": len(rows), "rows": rows, "summary": summary,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("fixture", "pools", "replay", "database", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--stride", default=4, type=int)
    p.add_argument("--offset", default=0, type=int)
    p.add_argument("--threads", default=1, type=int)
    a = p.parse_args()
    assert 0 <= a.offset < a.stride
    torch.set_num_threads(a.threads)
    a.output.write_text(json.dumps(run(a), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
