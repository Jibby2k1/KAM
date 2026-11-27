#!/usr/bin/env python3
"""Audit exact stable top-k and diagnose restricting passage scans to likely documents."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_bright_passage_windows import load_records
from run_kam_bright_transfer import RRF_DEPTH, rrf
from run_kam_natural_feasibility import Encoder

SHORTLISTS = (128, 512, 2048)


def stable_topk(scores, k):
    values, ids = scores.topk(k + 1, dim=-1, sorted=True)
    at_boundary = torch.where(values[:, k] == values[:, k - 1])[0]
    selected = ids[:, :k].sort(dim=-1).values
    selected = selected.gather(1, scores.gather(1, selected).argsort(descending=True, stable=True))
    if at_boundary.numel():
        exact = scores.index_select(0, at_boundary).argsort(descending=True, stable=True)[:, :k]
        selected.index_copy_(0, at_boundary, exact)
    return selected, at_boundary.numel()


def exclusions(cases, device):
    ids = [(i, d) for i, case in enumerate(cases) for d in case["excluded"]]
    rows, docs = torch.tensor(ids, dtype=torch.long, device=device).reshape(-1, 2).T
    return rows, docs


def ranked(a):
    bundle = json.loads((a.inputs / "fixture.json").read_bytes())
    signature = digest(a.inputs / "fixture.json")
    revisions = json.loads((a.assets / "revisions.json").read_bytes())
    lex = json.loads(gzip.decompress(a.lexical.read_bytes()))
    prior = json.loads(gzip.decompress(a.reference.read_bytes()))
    assert prior["fixture_sha256"] == lex["fixture_sha256"] == signature
    cases = bundle["rows"]
    assert [r["id"] for r in prior["rows"]] == [r["id"] for r in lex["rows"]] == [r["id"] for r in cases]
    head_rows = load_records(a.head, "provenance", bundle, revisions, signature)
    extra = load_records(a.extra, "part", bundle, revisions, signature)
    head = torch.cat([r["vectors"] for r in head_rows]).cuda()
    windows = torch.cat([r["vectors"] for r in extra]).cuda()
    owner = torch.cat([r["owner"] for r in extra]).cuda()
    lex_ids = torch.tensor([r["ids"] for r in lex["rows"]], device="cuda")
    ref = [r["ranks"] for r in prior["rows"]]
    queries = Encoder(a.assets / "encoder", 128)([r["question"] for r in cases], "query")

    def scores(q, bans, mode):
        base = q @ head.T
        alternate = q @ windows.T
        base[bans] = -torch.inf
        visible = base.new_ones((len(q),), dtype=torch.int64) * len(owner)
        if mode in SHORTLISTS:
            active = torch.zeros_like(base, dtype=torch.bool)
            tops, _ = stable_topk(base, mode)
            active.scatter_(1, tops, True)
            active.scatter_(1, lex_ids, True)
            available = active.index_select(1, owner)
            visible = available.sum(dim=1)
            alternate = alternate.masked_fill(~available, -torch.inf)
        base.scatter_reduce_(1, owner.expand_as(alternate), alternate, reduce="amax", include_self=True)
        base[bans] = -torch.inf
        return base, visible

    def arrays(q, bans, method):
        full, _ = scores(q, bans, "global")
        if method == "sort":
            return full.argsort(descending=True, stable=True)[:, :RRF_DEPTH].tolist(), 0
        ids, ties = stable_topk(full, RRF_DEPTH)
        return ids.tolist(), ties

    bans = exclusions(cases, "cuda")
    normal, ties = arrays(queries, bans, "topk")
    assert normal == arrays(queries, bans, "sort")[0]
    globals_only = [rrf((rank, lex["rows"][i]["ids"]), (1., 1.)) for i, rank in enumerate(normal)]
    comparison = [{"id": case["id"], "ranks": {"hybrid_head": baseline["hybrid_head"],
                                              "hybrid_windows": found},
                   "window_pairs": {}} for case, baseline, found in zip(cases, ref, globals_only, strict=True)]
    archive_matches = sum(r["hybrid_windows"] == found for r, found in zip(ref, globals_only, strict=True))
    if a.require_archive:
        assert archive_matches == len(cases)
    for capacity in SHORTLISTS:
        matrix, count = scores(queries, bans, capacity)
        ids, _ = stable_topk(matrix, RRF_DEPTH)
        for i, (row, docs, n) in enumerate(zip(comparison, ids.tolist(), count.tolist(), strict=True)):
            label = f"shortlist_{capacity}"
            row["ranks"][label] = rrf((docs, lex["rows"][i]["ids"]), (1., 1.))
            row["window_pairs"][label] = n

    rng = Random(485921)
    results = {}
    for width in (1, 8, len(cases)):
        offsets = list(range(len(cases) - width + 1))
        starts = rng.sample(offsets, min(12, len(offsets)))
        if len(starts) == 1:
            starts *= 12
        samples = {m: [] for m in ("sort", "topk")}
        observed = Counter()
        for first in starts:
            query = queries[first:first + width]
            local_bans = exclusions(cases[first:first + width], "cuda")
            assert arrays(query, local_bans, "sort")[0] == arrays(query, local_bans, "topk")[0]
            for policy in rng.sample(list(samples), 2):
                torch.cuda.synchronize()
                start = perf_counter()
                _, ambiguous = arrays(query, local_bans, policy)
                torch.cuda.synchronize()
                samples[policy].append(1000 * (perf_counter() - start))
                observed[policy] += ambiguous
        results[width] = {"ms": samples, "p50_ms": {k: median(v) for k, v in samples.items()},
                          "tie_rows": dict(observed), "batches": starts}
    return {"fixture_sha256": signature, "reference_sha256": digest(a.reference), "device": torch.cuda.get_device_name(),
            "task": bundle["task"], "documents": len(head), "extra_windows": len(windows),
            "archive_matches": archive_matches, "within_device_exact": len(cases),
            "global_boundary_ties": ties, "timed_rank": results, "rows": comparison}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "lexical", "assets", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("head", "extra"):
        p.add_argument("--" + name, type=Path, nargs=4, required=True)
    p.add_argument("--require-archive", action="store_true")
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    with torch.inference_mode():
        values = ranked(a)
    a.output.write_bytes(gzip.compress(json.dumps(values, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
