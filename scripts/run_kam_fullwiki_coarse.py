#!/usr/bin/env python3
"""Build a spherical posting index and replay conditioned full-Wikipedia search."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch.nn.functional import normalize

from audit_kam_natural_data import digest
from run_kam_fullwiki_native_compare import FullWikiIndex
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts
from run_kam_sentence_verifier_fullwiki import pool_text

CENTERS = 2048
SAMPLE = 65_536
ASSIGNMENT_BLOCK = 8192
DEPTH = 64
RESCORE = 96
PROBES = (16, 64, 256)
ITERATIONS = 12


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def distance(matrix, directions):
    return torch.mm(matrix.half(), directions.half().T, out_dtype=torch.float32)


def assignment(samples, directions):
    return distance(samples, directions).argmax(dim=1)


@torch.inference_mode()
def build(a, index, manifest):
    eligible = torch.nonzero(index.mask).flatten()
    assert len(eligible) == index.meta["indexed"] > SAMPLE
    rng = torch.Generator().manual_seed(a.seed)
    selected = eligible[torch.randperm(len(eligible), generator=rng)[:SAMPLE].cuda()]
    training = normalize(index.bank[selected], dim=-1)
    centroids = training[torch.randperm(SAMPLE, generator=rng)[:CENTERS].cuda()].contiguous()
    torch.cuda.synchronize()
    trained = perf_counter()
    for _ in range(ITERATIONS):
        labels = torch.cat([assignment(piece, centroids) for piece in training.split(ASSIGNMENT_BLOCK)])
        sums = torch.zeros_like(centroids).index_add_(0, labels, training)
        counts = labels.bincount(minlength=CENTERS)
        centroids = torch.where(counts[:, None] > 0, normalize(sums, dim=-1), centroids)
    torch.cuda.synchronize()
    training_seconds = perf_counter() - trained

    started = perf_counter()
    labels = torch.cat([assignment(normalize(index.bank[ids], dim=-1), centroids)
                        for ids in eligible.split(ASSIGNMENT_BLOCK)])
    counts = labels.bincount(minlength=CENTERS)
    order = labels.argsort(stable=True)
    offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0))
    ordered = eligible[order].to(torch.int32).cpu()
    assert len(ordered) == int(offsets[-1]) == len(eligible)
    result = {"centroids": centroids.cpu(), "ordered_ids": ordered, "offsets": offsets.cpu(),
              "seed": a.seed, "iterations": ITERATIONS, "sample": SAMPLE, "partition_count": CENTERS,
              "index_sha256": digest(a.inputs / "lexical/completed.json"), "source_sha256": manifest["source_sha256"],
              "max_list": int(counts.max()), "empty_lists": int((counts == 0).sum()),
              "training_seconds": training_seconds, "assignment_seconds": perf_counter() - started,
              "device": torch.cuda.get_device_name(), "numeric": str(torch.__version__)}
    torch.save(result, a.output)


class Search:
    def __init__(self, index, partition):
        self.fp32 = index.bank
        self.full_half = self.fp32.half()
        self.mask = index.mask
        self.centroids = partition["centroids"].cuda()
        self.posting_ids = partition["ordered_ids"].cuda()
        self.offsets = partition["offsets"].tolist()
        self.offsets_gpu = partition["offsets"].cuda()
        self.max_list = partition["max_list"]
        assert self.offsets[-1] == len(self.posting_ids) and self.offsets[0] == 0
        assert torch.equal(self.posting_ids.long().sort().values, torch.nonzero(self.mask).flatten())
        self.posting_bank = self.full_half[self.posting_ids]
        self.resident_bytes = self.centroids.nbytes + self.posting_ids.nbytes + self.posting_bank.nbytes

    def exact(self, q, exclude):
        scores = torch.mv(self.fp32, q[0]).masked_fill_(~self.mask, -torch.inf)
        scores[exclude] = -torch.inf
        return scores.topk(DEPTH).indices, len(self.fp32)

    def rescore(self, q, shortlist):
        prec = torch.mv(self.fp32[shortlist], q[0])
        return shortlist[prec.topk(DEPTH).indices]

    def full_rescore(self, q, exclude):
        scores = distance(q, self.full_half)[0].masked_fill_(~self.mask, -torch.inf)
        if exclude is not None:
            scores[exclude] = -torch.inf
        return self.rescore(q, scores.topk(RESCORE).indices), len(self.full_half)

    def probe(self, q, exclude, probes):
        lists = (self.centroids @ q[0]).topk(probes).indices.tolist()
        ranges = [(self.offsets[i], self.offsets[i + 1]) for i in lists]
        assert sum(end - begin for begin, end in ranges) >= RESCORE
        ids = torch.cat([self.posting_ids[a:b] for a, b in ranges])
        bank = torch.cat([self.posting_bank[a:b] for a, b in ranges])
        scores = distance(q, bank)[0].masked_fill_(ids == exclude, -torch.inf)
        return self.rescore(q, ids[scores.topk(RESCORE).indices]), len(ids)

    def fused_probe(self, q, exclude, probes):
        from kam_coarse_scores import shortlist

        lists = (self.centroids @ q[0]).topk(probes).indices
        ids = shortlist(self.posting_bank, self.posting_ids, self.offsets_gpu, lists,
                        q[0].half(), exclude, RESCORE, self.max_list)
        scanned = (self.offsets_gpu[lists + 1] - self.offsets_gpu[lists]).sum()
        return self.rescore(q, ids), scanned


@torch.inference_mode()
def replay(a, index, cfg, manifest):
    init = perf_counter()
    fixture_bytes = a.fixture.read_bytes()
    if a.fixture.suffix == ".gz":
        fixture_bytes = gzip.decompress(fixture_bytes)
    source = json.loads(fixture_bytes)
    fixture_hash = sha256(fixture_bytes).hexdigest()
    original = read(a.pools)
    archive = read(a.reference)
    verify = digest(a.inputs / "lexical/completed.json")
    assert manifest["source_sha256"] == source["source_sha256"]
    assert original["index_sha256"] == verify and original["fixture_sha256"] == fixture_hash
    partition = torch.load(a.partition, weights_only=True, map_location="cpu")
    assert partition["index_sha256"] == verify and partition["source_sha256"] == manifest["source_sha256"]
    assert all(item["fixture_sha256"] == fixture_hash and item["pools_sha256"] == digest(a.pools)
               and item["index_sha256"] == verify for item in archive)
    anchor = {q["id"]: q for item in archive for q in item["rows"]}
    cases = source["rows"]["test"]
    assert len(cases) == 1024 and len(anchor) == len(original["rows"]) == len(cases)
    cases = cases[a.shard::a.shards]
    engine = Search(index, partition)
    methods = {"exact": engine.exact, "full_rescore": engine.full_rescore}
    methods.update({f"probe_{p}": (lambda vec, i, p=p: engine.probe(vec, i, p)) for p in PROBES})
    if a.fused:
        methods.update({f"fused_{p}": (lambda vec, i, p=p: engine.fused_probe(vec, i, p)) for p in (64, 256)})
    adapter, _ = restore_checkpoint(a.baseline, a.config, cfg, None, None, weights_sha256=a.baseline_sha256)
    enc = Encoder(a.assets / "encoder", 128)
    queries = []
    for start in range(0, len(cases), 12):
        chunk = cases[start:start+12]
        raw = adapter(enc([q["question"] for q in chunk], "query"))
        contexts, ids = [], []
        for item in chunk:
            qid = item["id"]
            first = anchor[qid]["first"]
            prior = original["rows"][qid]
            assert len(prior) == DEPTH and first in prior
            example = pool_text(index.db, item["question"], prior)
            first_text = candidate_texts(example)[prior.index(first)]
            contexts.append(f"{item['question']} First retrieved document: {first_text}")
            ids.append(first)
        cond = (raw + adapter(enc(contexts, "query"))) * 0.5
        queries.extend((q.unsqueeze(0), first) for q, first in zip(cond, ids, strict=True))
    torch.cuda.synchronize()
    init = perf_counter() - init

    file = a.partition
    partition_hash = digest(file)
    rng = Random(47581 + a.shard)
    for query, excluded in queries[:8]:
        for method in methods.values():
            method(query, excluded)[0].tolist()
    rows = []
    for i in rng.sample(range(len(cases)), len(cases)):
        q, (vector, excluded) = cases[i], queries[i]
        data = {}
        for name in rng.sample(list(methods), len(methods)):
            torch.cuda.synchronize()
            start = perf_counter()
            result, scanned = methods[name](vector, excluded)
            items = result.tolist()
            latency = perf_counter() - start
            assert len(items) == len(set(items)) == DEPTH and excluded not in items
            data[name] = {"ranks": items, "scanned": int(scanned), "seconds": latency}
        rows.append({"id": q["id"], "methods": data})
    result = {"fixture_sha256": fixture_hash, "pools_sha256": digest(a.pools),
              "archive_sha256": digest(a.reference), "partition_sha256": partition_hash,
              "index_sha256": verify, "device": torch.cuda.get_device_name(), "numeric": str(torch.__version__),
              "shard": a.shard, "shards": a.shards, "init_seconds": init, "indexed": index.meta["indexed"],
              "coarse_index_gpu_bytes": engine.resident_bytes, "full_half_bytes": engine.full_half.nbytes,
              "full_exact_bytes": engine.fp32.nbytes, "rows": rows}
    a.output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("build", "replay"))
    for name in ("inputs", "config", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in ("assets", "partition", "fixture", "pools", "reference", "baseline"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--baseline-sha256")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=862002)
    p.add_argument("--fused", action="store_true")
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True)
    cfg = read(a.config) | {"dense_rescore": False}
    manifest = read(a.inputs / "manifest.json")
    index = FullWikiIndex(a.inputs, manifest, cfg)
    if a.mode == "build":
        build(a, index, manifest)
    else:
        replay(a, index, cfg, manifest)


if __name__ == "__main__":
    main()
